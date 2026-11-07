#!/usr/bin/env python
"""The learning-rate schedule, as one function the trainer and its test both call.

WHY IT IS ITS OWN FILE. It used to be eight lines inlined in the training loop, which meant the
only way to know what lr epoch 50 would see was to run fifty epochs. The schedule is one of the
three interventions this arm is testing, so it needs to be assertable before the run, not after:
run_logging/test_lr_schedule.py pins the lr at the epochs the launch script names.

THE BUG THAT MOTIVATED THE EXTRACTION was not in this arithmetic -- it was in the HORIZON passed
to it. phase_k_train derived steps-per-epoch from len(flat) // world, but build_epoch takes
floor(n_seeds / world) WHOLE seed-groups per structure, so at ten seeds and world 3 one seed per
structure sits out. That counted 205 steps per epoch against an actual 184, and an 11% stretch in
the step horizon moved "1e-4 by epoch 50" to epoch 55.7, leaving lr about 8x its intended value at
epoch 50. The arithmetic here is unchanged from what shipped; `steps_per_epoch` is now the thing
under test.
"""
import math


def parse_lr_steps(spec):
    """"20:1e-3,40:1e-4" -> [(20.0, 1e-3), (40.0, 1e-4)], sorted by epoch.

    Raises ValueError rather than guessing, because a mis-parsed schedule is invisible: the run
    trains at the wrong lr for twenty epochs and nothing in the log says so.
    """
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"--lr-steps entry {part!r} is not EPOCH:LR")
        e, v = part.split(":", 1)
        ep, val = float(e), float(v)
        if ep < 0:
            raise ValueError(f"--lr-steps epoch {ep} is negative")
        if val <= 0:
            raise ValueError(f"--lr-steps lr {val} is not positive")
        out.append((ep, val))
    out.sort(key=lambda t: t[0])
    for (e0, _), (e1, _) in zip(out, out[1:]):
        if e0 == e1:
            raise ValueError(f"--lr-steps names epoch {e0} twice")
    return out


def lr_at_step(step, lr, lr_final, warmup_steps, steps_per_epoch, epochs,
               decay_start_epoch=0.0, decay_end_epoch=0.0, shape="cosine", lr_steps=()):
    """The lr this optimiser step runs at. `step` is 0-based and resume restores it.

    Defaults reproduce the historical schedule exactly: warm up over warmup_steps, then either
    hold lr flat (lr_final <= 0) or cosine from the end of warmup to the last expected step.

    lr_steps, when given, is a STAIRCASE and takes precedence over every decay flag: the lr is
    `lr` until the first milestone epoch, then each milestone's value until the next. It is a
    different object from cosine/exponential, which interpolate continuously -- a staircase holds
    a value and then drops, which is what "drop to 1e-3 at epoch 20" asks for and what neither
    shape can express. Warmup still applies at the very start, so the first steps ramp rather than
    beginning at full lr.
    """
    horizon = float(steps_per_epoch) * max(1.0, float(epochs))
    cur = lr * min(1.0, (step + 1) / max(1, warmup_steps))
    if lr_steps:
        # STAIRCASE. spe comes from the same horizon the other shapes use, so a schedule pinned in
        # a test is pinned against the trainer's own step geometry and not an idealised one.
        spe = horizon / max(1e-9, float(epochs))
        ep_now = step / max(1e-9, spe)
        val = lr
        for ep_at, v in lr_steps:
            if ep_now >= ep_at:
                val = v
        # warmup only ever LOWERS the first steps; it cannot raise a later milestone back up.
        return min(val, cur) if step < warmup_steps else val
    if lr_final <= 0.0 or horizon <= warmup_steps or step < warmup_steps:
        return cur
    spe = horizon / max(1e-9, float(epochs))
    d0 = warmup_steps if decay_start_epoch <= 0.0 else max(warmup_steps,
                                                           decay_start_epoch * spe)
    d1 = horizon if decay_end_epoch <= 0.0 else decay_end_epoch * spe
    if step < d0:
        return lr                                  # still in the flat hold
    if d1 <= d0:
        return lr_final                            # degenerate window: land immediately
    p = min(1.0, (step - d0) / float(d1 - d0))
    if shape == "exponential":
        return lr * (lr_final / lr) ** p
    return lr_final + 0.5 * (lr - lr_final) * (1.0 + math.cos(math.pi * p))


def units_per_rank(units_by_pdb, world, cross_seed=False, n_units=None):
    """Units ONE RANK walks in one epoch, as the epoch builder groups them.

    build_epoch takes floor(n_seeds / world) WHOLE seed-groups of each structure, so the remainder
    -- one seed per structure at ten seeds and world 3 -- sits out, reseeded each epoch. That makes
    the per-rank count a SUM OF FLOORS, not one division of the total: 368 x floor(10/3) = 1,104,
    where len(flat) // world gives 1,226 and counts a seed that never trains.
    """
    if cross_seed:
        return int(n_units if n_units is not None
                   else sum(units_by_pdb.values())) // max(1, world)
    return sum(v // max(1, world) for v in units_by_pdb.values())


def deal_round_robin(lists):
    """One cross-seed step's units, dealt round-robin across its structures.

    THE ORDER OF THE RETURNED LIST IS THE RANK ASSIGNMENT, because phase_k_train splits a step
    into CONTIGUOUS per-rank slices. Dealing one unit from each structure in turn means a
    contiguous slice of length len(units)/world holds len(s)/world units of EVERY structure s,
    which is exact whenever each depth divides `world`.

    WHAT IT REPLACED was a global shuffle of the step's units, which gave each rank a random
    subset -- for a step of two 15-seed structures, anything from 10+0 to 5+5. Per-unit memory and
    time go as tokens^2, so a lopsided split makes every rank wait on the slowest at the step's
    all-reduce.

    `lists` is a list of per-structure unit lists and is not mutated. The result is a permutation
    of their concatenation: same units, same count, different order. Shuffle `lists` beforehand if
    the rotation should not always be led by the same structure.
    """
    q = [list(x) for x in lists]
    out = []
    while any(q):
        for one in q:
            if one:
                out.append(one.pop(0))
    return out


def pack_cross_seed_steps(depth, order, structs_per_step, preserve_order=False):
    """ONE packing, used by the epoch builder AND by the horizon. in: {pdb: n_units}, the epoch's
    structure order, S.  out: (list of chunks of pdb, dropped pdbs, unit budget per step).

    WHY THIS IS A FUNCTION AND NOT TWO. build_epoch_cross_seed packed units inline and this module
    predicted the step count with len(units_by_pdb) // S. That was right while a step was S whole
    structures; it stopped being right the moment a step became a constant number of UNITS, and the
    two drifted silently -- 853 // 2 = 426 claimed against 232 actually built, a 1.84x horizon
    error. A cosine decay configured for epochs 20..50 would have run 36.7..91.8, leaving lr eight
    times its intended value at epoch 50. That is the THIRD recurrence of this bug class in this
    file's history, which is why the prediction is now the implementation rather than a model of it.

    FIRST-FIT DECREASING, so the step count is a function of the DEPTH HISTOGRAM ALONE and not of
    the epoch's shuffle. Plain first-fit over a shuffled order gave 220..229 steps on the real
    437x15 + 416x1 population depending on the seed, so there was no constant steps-per-epoch to
    calibrate a schedule against even in principle. Sorting by descending depth is stable, so a
    caller that shuffles first keeps randomness WITHIN each depth class -- which structures share a
    step still varies per epoch -- while the count does not. It also wastes less: 11 units dropped
    against 19-35 structures.
    """
    S = max(1, int(structs_per_step))
    order = [p for p in order if depth.get(p)]
    if not order:
        return [], [], 0
    U = S * max(depth[p] for p in order)
    # preserve_order IS FOR --epoch-order manifest/tokens, WHERE THE ORDER IS THE POINT.
    # Raised in review and correct: sorting by depth overrides a size-ascending token order, and a
    # memory probe reads its threshold off exactly that order -- so the descending sort would have
    # quietly made --epoch-order tokens mean something else. It is safe to skip the sort there
    # because those modes do not reshuffle: the order is fixed across epochs by design, so the step
    # count is already stable without it. The sort exists only to stabilise the SHUFFLED modes.
    pend = list(order) if preserve_order else sorted(order, key=lambda p: -depth[p])
    steps, dropped = [], []
    while pend:
        room, chunk = U, []
        i = 0
        while i < len(pend) and room:
            p = pend[i]
            if depth[p] <= room:
                chunk.append(p)
                room -= depth[p]
                pend.pop(i)
            else:
                i += 1
        if room:
            # A SHORT FINAL STEP IS DROPPED, not padded: every downstream site reads the step's
            # unit count as a per-epoch constant.
            dropped = list(chunk) + list(pend)
            break
        steps.append(chunk)
    return steps, dropped, U


def steps_per_epoch(units_by_pdb, world, nsample, accum_local, cross_seed=False, n_units=None,
                    structs_per_step=0, preserve_order=False):
    """Optimiser steps in one epoch, DERIVED FROM THE EPOCH BUILDER'S OWN GROUPING.

    build_epoch walks floor(n_seeds / world) whole seed-groups of each structure -- the remainder
    sits out, reseeded each epoch -- so the per-rank unit count is a sum of floors, not a single
    division of the total. build_epoch_cross_seed keeps every seed and rotates the split, so there
    the total divided by world is right.

    CROSS-SEED WITH structs_per_step DOES NOT USE accum_local AT ALL, and missing that stretched a
    live run's schedule by 1.68x. build_epoch_cross_seed chunks the SHUFFLED STRUCTURE LIST into
    groups of --rank-structs-per-step and makes each chunk one optimiser step, because a pose's
    coefficient needs every other pose of its interface in the same step. So the step count is
    floor(n_structures / S) -- 122 at 368 structures and S=3 -- and dividing poses by accum_local
    instead gave 205. The decay would then have reached 1e-4 near epoch 84 rather than 50. This is
    the same class of error the function was written to fix (205 assumed against 184 actual),
    recurring in the branch the first fix did not cover, and it is why structs_per_step is now a
    parameter rather than inferred.

    A SHORT FINAL CHUNK IS DROPPED by the builder, not padded, so this is a floor and not a
    ceiling: at 368 and S=3 the last 2 structures do not form a step.
    """
    if cross_seed and structs_per_step and structs_per_step > 0:
        # THE PACKER ITSELF, not a formula that models it. See pack_cross_seed_steps.
        _st, _dr, _U = pack_cross_seed_steps(dict(units_by_pdb), list(units_by_pdb),
                                             structs_per_step, preserve_order=preserve_order)
        return max(1, len(_st))
    per_rank = units_per_rank(units_by_pdb, world, cross_seed, n_units)
    # CEILING, NOT FLOOR PLUS ONE. The loop flushes a partial accumulation at the epoch
    # boundary, but only when one exists -- and at 368 structures the division is exact
    # (1,104 units x 5 poses / 30 = 184), so an unconditional +1 claims 185 and stretches
    # the horizon by a further 0.5%. That was the shipped formula: the same class of error
    # as the one this function exists to fix, just smaller.
    return int(math.ceil(per_rank * nsample / max(1, accum_local)))
