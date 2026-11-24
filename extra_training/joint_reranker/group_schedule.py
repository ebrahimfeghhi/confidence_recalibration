"""Deterministic, full-coverage interface batches for a joint pose reranker.

Unlike the legacy epoch planners, this never splits an attention group across
ranks, and never assumes a constant number of pose/seed units per step.

Input candidates come from the declared (pdb, seed, sample, interface) manifest;
pose_index indexes its live tensor row, while sample_index joins the label.
DockQ is deliberately absent from the scheduling API, so no native reference
can accidentally leak into inference feature selection.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PoseRef:
    pdb: str
    interface: str
    seed: str
    pose_index: int
    sample_index: int
    crop_tokens: int
    # A key to the existing trunk/labels, not an extra copy of tensors.
    unit_key: str = ""

    def __post_init__(self) -> None:
        if not self.pdb or not self.interface or not self.seed:
            raise ValueError("pdb, interface and seed must be nonempty")
        if self.pose_index < 0 or self.sample_index < 0 or self.crop_tokens <= 0:
            raise ValueError("pose/sample indices must be nonnegative and crop positive")

    @property
    def identity(self) -> tuple[str, str, str, int]:
        return self.pdb, self.interface, self.seed, self.sample_index


@dataclass(frozen=True)
class InterfaceBatch:
    pdb: str
    interface: str
    groups: tuple[tuple[PoseRef, ...], ...]

    @property
    def n_poses(self) -> int:
        return sum(len(group) for group in self.groups)

    @property
    def estimated_work(self) -> int:
        return sum(sum(p.crop_tokens ** 2 for p in group) for group in self.groups)


@dataclass(frozen=True)
class OptimizerStep:
    """All ranks see this *same* step. None means idle but still all-reduce."""
    per_rank: tuple[InterfaceBatch | None, ...]

    @property
    def n_poses(self) -> int:
        return sum(b.n_poses for b in self.per_rank if b is not None)

    @property
    def n_interfaces(self) -> int:
        return sum(b is not None for b in self.per_rank)

    def loss_coefficients(self, rank: int, group_index: int) -> tuple[float, float]:
        """(PAE, ranking) scales for the group on this rank.

        With manual ReduceOp.SUM across ranks, sum of
          (group PAE mean * PAE scale) -> GLOBAL pose mean;
          (group listwise loss * rank scale) -> GLOBAL interface mean,
        where an interface's multiple <=K groups are averaged first.
        Empty/uncontrasted groups still count for both denominators.
        """
        batch = self.per_rank[rank]
        if batch is None:
            raise ValueError("idle rank has no group")
        if not 0 <= group_index < len(batch.groups):
            raise IndexError("group_index out of range")
        k = len(batch.groups[group_index])
        return k / self.n_poses, 1.0 / (self.n_interfaces * len(batch.groups))


def scaled_group_loss(step: OptimizerStep, rank: int, group_index: int,
                      joint_result, *, alpha: float):
    """Differentiable per-group objective for legacy manual SUM grad reduction.

    IMPORTANT: use joint_result.pae and .ranking, *not* .total -- .total
    already contains alpha and a mean over only the current K poses, and is
    not correctly normalized for three ranks or variable group sizes.
    """
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    pae_scale, rank_scale = step.loss_coefficients(rank, group_index)
    return joint_result.pae * pae_scale + joint_result.ranking * (alpha * rank_scale)


def _chunks_without_one_tail(items: list[PoseRef], cap: int) -> tuple[tuple[PoseRef, ...], ...]:
    """Use all poses exactly once; avoid K=1 tails whenever feasible."""
    chunks = [items[i:i + cap] for i in range(0, len(items), cap)]
    if len(chunks) > 1 and len(chunks[-1]) == 1 and len(chunks[-2]) > 2:
        chunks[-1].insert(0, chunks[-2].pop())
    return tuple(tuple(group) for group in chunks)


def plan_epoch(
    candidates: Sequence[PoseRef],
    safe_k_by_crop: Mapping[int, int],
    *,
    world: int,
    seed: int,
    epoch: int,
) -> tuple[OptimizerStep, ...]:
    """One *whole interface* per rank per optimizer step, streamed <=K groups.

    For an interface with 75 poses and a cap of 20 this builds 20+20+20+15.
    The four sets run sequentially under one rank's gradient accumulation,
    and give one ranking-objective weight to the interface in total.

    The global step size depends on which interfaces are scheduled together;
    the returned loss coefficients must be read *per step* rather than from
    steps[0] or a constant from the legacy unit-count planner.

    Explicit `None` final-rank entries are safe only with the repository's
    MANUAL gradient ReduceOp.SUM (zero-fill all missing grads at the collective).
    Do not substitute a standard DDP forward without handling idle ranks.
    """
    if world <= 0:
        raise ValueError("world must be positive")
    if not candidates:
        raise ValueError("cannot schedule an empty candidate manifest")
    caps = dict(safe_k_by_crop)
    if any(k <= 0 or c <= 0 for c, k in caps.items()):
        raise ValueError("every crop/capacity must be positive")
    rng = random.Random(seed + epoch * 1000003)
    by_interface: dict[tuple[str, str], list[PoseRef]] = defaultdict(list)
    seen: set[tuple[str, str, str, int]] = set()
    for p in candidates:
        if p.crop_tokens not in caps:
            raise ValueError(f"crop {p.crop_tokens} has no verified K capacity")
        if p.identity in seen:
            raise ValueError(f"duplicate candidate identity {p.identity}")
        seen.add(p.identity)
        by_interface[p.pdb, p.interface].append(p)

    bundles: list[InterfaceBatch] = []
    for (pdb, interface), poses in sorted(by_interface.items()):
        # Sorting before shuffle decouples the plan from input manifest order.
        by_crop: dict[int, list[PoseRef]] = defaultdict(list)
        for pose in sorted(poses, key=lambda p: (p.crop_tokens, p.seed, p.sample_index)):
            by_crop[pose.crop_tokens].append(pose)
        groups: list[tuple[PoseRef, ...]] = []
        for crop, crop_poses in sorted(by_crop.items()):
            rng.shuffle(crop_poses)
            groups.extend(_chunks_without_one_tail(crop_poses, caps[crop]))
        rng.shuffle(groups)
        bundles.append(InterfaceBatch(pdb, interface, tuple(groups)))

    # Similar-cost bundles land in the same optimizer step, reducing rank
    # straggler time; shuffle their rank positions each epoch for fairness.
    rng.shuffle(bundles)
    bundles.sort(key=lambda b: -b.estimated_work)  # stable: ties remain shuffled
    # Bucket similarly expensive interfaces first to reduce GPU stragglers,
    # then shuffle the STEP buckets: a global descending-cost epoch would
    # bias the optimizer by always showing large crops at high early LR.
    cost_buckets = [bundles[i:i + world] for i in range(0, len(bundles), world)]
    rng.shuffle(cost_buckets)
    steps: list[OptimizerStep] = []
    for block in cost_buckets:
        rng.shuffle(block)
        per_rank: tuple[InterfaceBatch | None, ...] = tuple(
            block[r] if r < len(block) else None for r in range(world)
        )
        steps.append(OptimizerStep(per_rank))

    # Fail closed on coverage. Grouping by interface intentionally duplicates
    # the underlying pose across DIFFERENT scored interfaces, not within one.
    got = [p.identity for step in steps for b in step.per_rank if b is not None
           for group in b.groups for p in group]
    if len(got) != len(seen) or set(got) != seen:
        raise AssertionError("interface scheduler duplicated or dropped candidate identities")
    return tuple(steps)
