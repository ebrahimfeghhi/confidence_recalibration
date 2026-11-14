"""Hold the FROZEN PREFIX of the confidence head for the span of ONE optimiser step, so pass B
does not recompute what pass A already computed.

WHAT THE PREFIX IS, AND WHY THE CUT IS WHERE IT IS. Under the default arm
(INBLOCK=1 LEAN_SINGLE=1 in shell_scripts/launch_default_arm.sh) --train-only leaves exactly 28
head tensors trainable: block 3's tri_att_start, tri_att_end and pair_transition, the four
pose_inblock_* projections, and linear_no_bias_pae / pae_ln. EVERYTHING ELSE IS FROZEN -- and that
includes block 3's own tri_mul_out and tri_mul_in. The deepest point at which `z` is still a
function of frozen parameters alone is therefore not the stack's input and not block 3's input, it
is INSIDE block 3, after tri_mul_in and before the pose injection. That is the exact point
phases/pose_inblock.py already injects at, and for the same reason: it is where the backward graph
begins. So the cut and the injection are the same boundary seen from two sides.

    native injection   linear_no_bias_s1/s2 -> z_init, linear_no_bias_d(one_hot(cdist(x))),
                       linear_no_bias_d_wo_onehot(cdist(x))             FROZEN
    blocks 0, 1, 2     tri_mul_out, tri_mul_in, tri_att_start, tri_att_end, pair_transition
                                                                        FROZEN
    block 3            tri_mul_out, tri_mul_in                          FROZEN
    ================== THE CUT: `z` here is the cached value ===========================
    block 3            + pose_inblock_d(one_hot(d)) + pose_inblock_d_wo_onehot(d)   TRAINABLE
                       tri_att_start, tri_att_end, pair_transition      TRAINABLE
    readout            pae_ln (+ the pae injection), linear_no_bias_pae TRAINABLE

WHY IT IS WORTH ANYTHING. phases/phase_k_train.py runs TWO passes over every pose of every step:
pass A is a no_grad forward whose scores the relative-iTM coefficients need (a pose's coefficient
reads every other pose of its interface, so all scores must exist before any backward), and pass B
rebuilds each pose and backwards it. Every pose is forwarded twice and the frozen prefix runs in
BOTH. Measured on mqird6 epoch 6, rank 0, of a 222.1 min epoch: gpu_forward 69.7 min over 23,000
calls, i.e. ~34.9 min per pass. The prefix is three whole blocks plus two of block 3's five pair
operations, so filling in pass A and hitting in pass B removes most of one pass's forward.

ONLY z IS CACHED, AND THAT IS A CONSEQUENCE OF --lean-single. pose_inblock's mirror updates the
single track under `if blk.c_s > 0:` and lean_readouts.drop_single_pathway sets c_s=0 on every
block, so `s` passes through the whole stack untouched and the suffix does not need it. The pose
geometry (`oh`, `dd`) is an O(N^2) cdist and one-hot that the pose_inblock wrapper recomputes per
call anyway; recomputing it is cheaper than holding an [N, N, 39] fp32 tensor per pose, so it is
NOT cached. `pair_mask` is not cached either: the caller passes it on every call (None, in this
arm) and it is folded into the key instead, so a different mask is a MISS and not a wrong answer.

THE CACHE IS PER POSE AND NEVER SHARED BETWEEN POSES. The prefix depends on the pose: the native
injection is a cdist over that pose's coordinates. Two poses of the same unit have the same trunk
and DIFFERENT z at the cut.

WHY IT IS IN RAM AND NOT ON DISK, which is where issue #8 asked for it. z is [N, N, 256] in bf16 --
bf16 because the trunk's z_trunk is stored as bf16 and autocast keeps it there. At the median 583
tokens that is 174 MB per pose; 11,500 poses per rank per epoch over 3 ranks is ~6 TB per epoch
against a 3,072 GiB hard quota with no grace. Persistent caching is infeasible by two orders of
magnitude, and the reuse distance does not need it: a pose is read twice within ONE step and never
again, because the epoch's poses are distinct. Issue #8 step 7 allows exactly this
("If disk read cost or storage makes persistent cache unattractive, propose RAM/CPU staging").

THE HOST, NOT THE DEVICE. 50 poses per rank per step at 174 MB is ~8.7 GB typical, ~26 GB at the
p90 of 1,004 tokens and ~70 GB if every unit of a step were at the 1,650-token cap. The cards run
at ~76 of 80 GB already, so a device-resident cache would OOM; the host pays a D2H on the fill and
an H2D on the hit instead, ~10 ms each at the median, which is ~4 min of a 222 min epoch against
the ~30 it removes.

AND IT IS BYTE-CAPPED, WITH THE RESERVATION TAKEN UNDER THE LOCK. --prefix-cache-gb defaults to 0
(off). The discipline is copied from the trunk cache in phase_k_train.py (_tr_cache / _tr_inflight
/ _tr_resv / _tr_put / _tr_reset), including the bug review found there: a cap that counts only
ADMITTED bytes does not bound memory, because the allocation happens before the admission. The
bytes are reserved at the instant the check passes and released in a `finally` AND on the error
path, so the cap bounds admitted + in-flight, which is what RSS actually is.

PURELY ADVISORY. A pose that does not fit the cap, a pose whose key was never set, a pose whose
cached entry does not match the shape or dtype it is asked for -- every one of those recomputes the
prefix. Correctness never depends on a hit, and `prefix_cache.py` removed entirely leaves the arm
numerically unchanged.

WHAT IS *NOT* SKIPPED ON A HIT, stated because the honest number matters. The cut is inside block 3,
but ConfidenceHead.forward and lean_readouts' mirror run BEFORE the stack is reached, and this
module does not patch either of them: z_init (two O(N) linears and an O(N^2) broadcast add), the
fp32 cdist, the [N, N, 39] one-hot and the two native pose linears still run on a hit, and their
result is discarded when the cut replaces `z`. That is roughly 7 GFLOP against the prefix's ~400 at
the median -- about 2% -- and skipping it would mean mirroring a third pinned function. It is left
running deliberately: on a hit the suffix is literally the same code objects, called with the same
arguments, as on a miss, which is the strongest correctness argument available and worth 2%.
"""
import hashlib
import threading

import torch

# BUMP THIS WHENEVER THE CUT, THE SET OF CACHED TENSORS, OR THEIR MEANING CHANGES. It is part of
# every key, so an entry written by older code can only MISS.
SCHEMA = "prefix-cache/1"

# THE HEAD PARAMETERS THE PREFIX READS THAT ARE NOT IN A PAIRFORMER BLOCK. Checked for
# requires_grad=False at install and again at every step boundary, and folded into the value digest
# below. input_strunk_ln is deliberately ABSENT: it is applied to s_trunk, and under --lean-single
# `s` reaches nothing that `z` depends on, so its weights cannot change the cached value.
PREFIX_HEAD_PARAMS = ("linear_no_bias_s1", "linear_no_bias_s2",
                      "linear_no_bias_d", "linear_no_bias_d_wo_onehot",
                      "lower_bins", "upper_bins")

# THE SUBMODULES OF THE CUT BLOCK THAT ARE STILL IN THE PREFIX. Everything else in that block is
# downstream of the injection.
PREFIX_CUT_BLOCK_SUBS = ("tri_mul_out", "tri_mul_in")

_LOCK = threading.Lock()
_S = {
    "on": False,
    "cap": 0,
    "bytes": 0,
    "resv": 0,
    "cache": {},        # full key -> host tensor at the cut
    "inflight": {},     # full key -> reserved bytes
    "gen": 0,
    "sig": None,
    "key": None,        # the pose the next forward is for, or None
    "fill": False,      # may this pose's prefix be offered to the cache?
    "head": None,
    "block_index": -1,
    "prefix_params": [],
    "prefix_names": set(),
    "dev": None,
    "n_tok": -1,
    "c_z": -1,
    "digest": None,
    "stat": {},
}


def _zero_stats():
    return {"hit": 0, "miss": 0, "put": 0, "full": 0, "nokey": 0, "shape": 0, "err": 0,
            "h2d_bytes": 0, "d2h_bytes": 0}


_S["stat"] = _zero_stats()


def _prefix_param_names(head, block_index):
    """Every head parameter the cached activation is a function of, by name.

    DERIVED FROM THE MODULE TREE, not written out. A hand-written list goes stale the moment a
    block gains a tensor, and the whole guarantee this module rests on is that none of these
    moves.
    """
    names = []
    for n, _p in head.named_parameters():
        if any(n == a or n.startswith(a + ".") for a in PREFIX_HEAD_PARAMS):
            names.append(n)
            continue
        if not n.startswith("pairformer_stack.blocks."):
            continue
        rest = n[len("pairformer_stack.blocks."):]
        bi, _, tail = rest.partition(".")
        if not bi.isdigit():
            continue
        bi = int(bi)
        if bi < block_index:
            names.append(n)
        elif bi == block_index and any(tail.startswith(s + ".") for s in PREFIX_CUT_BLOCK_SUBS):
            names.append(n)
    return names


def _digest(params):
    """A single scalar over every prefix parameter, for ONE device synchronisation.

    NOT A HASH OF THE BYTES. The point is to notice that a tensor the cache assumes is frozen has
    moved, and that has to be cheap enough to re-check at every one of 230 step boundaries. One
    stacked sum is one reduction and one sync; per-tensor `float(p.sum())` would be ~120 syncs a
    step. A sum can collide in principle and cannot in the case this guards against, which is an
    optimiser or a weight load touching a frozen tensor.
    """
    if not params:
        return "0"
    with torch.no_grad():
        tot = torch.stack([p.detach().float().sum() for p in params]).sum()
    return f"{float(tot):.17g}"


def _pm_desc(pair_mask):
    """A key component for pair_mask. Cheap when it is None, which is what this arm passes.

    CONTENT, NOT JUST SHAPE, WHEN IT IS A TENSOR. The prefix is a function of the mask, so two
    different masks of the same shape must not share an entry. That costs a reduction and a sync,
    which is why it only happens on the path this arm never takes.
    """
    if pair_mask is None:
        return "none"
    if not torch.is_tensor(pair_mask):
        return f"other:{type(pair_mask).__name__}"
    with torch.no_grad():
        return (f"t:{tuple(pair_mask.shape)}:{pair_mask.dtype}:"
                f"{float(pair_mask.detach().float().sum()):.17g}")


def _full_key(key, pair_mask):
    return (SCHEMA, _S["sig"], key, _pm_desc(pair_mask))


def _shape_ok(z):
    """Is this entry the shape and dtype the active pose's suffix will consume?

    [..., N_token, N_token, c_z] is the pair representation's shape everywhere in this head, and
    N_token comes from the caller via set_pose rather than being inferred, because the only other
    source of it on a hit is the cached tensor itself -- which cannot validate itself.

    DTYPE IS NOT CHECKED HERE, DELIBERATELY. The expected dtype at the cut is whatever the
    uncached computation would have produced, and the only way to know that is to run it. The
    fill and the hit are the same pose under the same code in the same step, so they agree by
    construction, and the cached tensor is stored at the dtype it was computed at -- never
    downcast, which would make a hit differ from a miss.
    """
    n = _S["n_tok"]
    return (torch.is_tensor(z) and z.dim() >= 3 and z.shape[-1] == _S["c_z"]
            and z.shape[-2] == n and z.shape[-3] == n)


def _put(full, z):
    """Offer the activation at the cut to the cache. Refused when full, which costs only the saving.

    IDEMPOTENT ON PURPOSE. With blocks_per_ckpt=1 the cut block's forward is RE-RUN inside backward,
    so this is called twice for one pose on a miss in pass B. The second call must not double-charge
    the cap, and it must not pay a second D2H.
    """
    nb = int(z.numel()) * int(z.element_size())
    with _LOCK:
        gen = _S["gen"]
        if full in _S["cache"] or full in _S["inflight"]:
            return
        if _S["bytes"] + _S["resv"] + nb > _S["cap"]:
            _S["stat"]["full"] += 1
            return
        # RESERVED BEFORE THE COPY, NOT AFTER IT. The host allocation below is the thing the cap is
        # supposed to bound; admitting first and checking later is the bug review found in the
        # trunk cache, where TRUNK_CACHE_GB bounded the dict while the process went 4x over it.
        _S["inflight"][full] = nb
        _S["resv"] += nb
    try:
        # DETACH FIRST. On the pass B miss path this runs with grad enabled and `z` carries no
        # graph (every parameter behind it is frozen), but storing a tensor that MIGHT carry one
        # would pin a graph for the span of a step.
        #
        # copy=True MATTERS EVEN ON CPU. Without it a CPU run would ALIAS the live activation, and
        # the cache would hand pass B a tensor some later op had written through -- the aliasing
        # trap phase_k_train's trunk cache avoids by refusing to run off cuda at all. copy=True
        # makes this correct on either device, which is also what lets the CPU test be a real test.
        host = z.detach().to("cpu", copy=True)
    except (RuntimeError, MemoryError):
        # ADVISORY: a host allocation that fails costs the saving and nothing else. Counted, not
        # swallowed silently.
        with _LOCK:
            _S["stat"]["err"] += 1
            _S["resv"] -= _S["inflight"].pop(full, 0)
        return
    except BaseException:
        # ANYTHING ELSE IS A BUG AND MUST SURFACE -- but the reservation is released first, or the
        # cap leaks and every later pose is refused.
        with _LOCK:
            _S["stat"]["err"] += 1
            _S["resv"] -= _S["inflight"].pop(full, 0)
        raise
    with _LOCK:
        try:
            # THE GENERATION IS CHECKED AFTER THE COPY TOO. step_reset() can land between the
            # reservation and here; charging the new step's cap with the previous step's tensor
            # would spend the budget on a pose nobody will ask for.
            if gen != _S["gen"] or full in _S["cache"]:
                return
            _S["cache"][full] = host
            _S["bytes"] += nb
            _S["stat"]["put"] += 1
            _S["stat"]["d2h_bytes"] += nb
        finally:
            _S["resv"] -= _S["inflight"].pop(full, 0)


def _plan(pair_mask=None, **_kw):
    """pose_inblock's cut hook: decide hit or fill for the pose that is about to run.

    Called ONCE per stack forward. Returns None to leave the forward exactly as it was.
    """
    if not _S["on"]:
        return None
    key = _S["key"]
    if key is None:
        # NOT AN ERROR. phase_k_eval and the held-out monitor build their own model and never set a
        # pose; so does any forward outside the training loop. No key means no cache, which is the
        # uncached path.
        _S["stat"]["nokey"] += 1
        return None
    full = _full_key(key, pair_mask)
    with _LOCK:
        ent = _S["cache"].get(full)
        _S["stat"]["hit" if ent is not None else "miss"] += 1
    if ent is not None and not _shape_ok(ent):
        # A MISS, NOT A WRONG ANSWER. The key already carries the token count, so this is the
        # belt to that braces: an entry whose shape or dtype is not what this pose's suffix is
        # about to consume is dropped and the prefix is recomputed. The alternative is a
        # broadcast that lines up by accident, which is how a cache returns a plausible number
        # for the wrong structure.
        with _LOCK:
            _S["stat"]["shape"] += 1
            _S["stat"]["hit"] -= 1
            _S["stat"]["miss"] += 1
            if _S["cache"].pop(full, None) is not None:
                _S["bytes"] -= int(ent.numel()) * int(ent.element_size())
        ent = None
    if ent is not None:
        dev = _S["dev"]
        nb = int(ent.numel()) * int(ent.element_size())
        _S["stat"]["h2d_bytes"] += nb
        return {"z": ent.to(dev) if dev is not None else ent}
    if not _S["fill"]:
        return None
    return {"put": lambda z: _put(full, z)}


# ---- THE TRAINER'S INTERFACE -------------------------------------------------------------------


def set_pose(key, fill, n_tok):
    """Name the pose the next head forward is for. Called per pose, from phase_k_train._one_pose.

    `key` identifies the pose WITHIN A STEP -- (pdb, arm_seed, pose_index) is enough, because the
    cache is dropped at every step boundary and a step holds each pose once. `n_tok` goes into the
    key as well, so a unit re-entered at a different token count (cropping, token dropout) cannot
    read an entry of the wrong shape; that is a MISS and a recompute.

    `fill` is True in pass A and False in pass B. A put in pass B would pay a D2H for a tensor
    nothing will read again, since the pose is done after its backward.
    """
    if not _S["on"]:
        return
    _S["key"] = (tuple(key), int(n_tok))
    _S["n_tok"] = int(n_tok)
    _S["fill"] = bool(fill)


def clear_pose():
    """Forget the active pose. Called in a `finally` around the forward.

    WITHOUT THIS, a forward that happens between poses -- anything the training loop does outside
    _one_pose -- would be keyed as the previous pose and could HIT on it. A key that is only set
    for the duration of one pose cannot do that.
    """
    _S["key"] = None
    _S["fill"] = False


def step_reset():
    """Retire the step's entries. Called at every step boundary and at the epoch boundary.

    AND RE-VERIFY THE PREMISE WHILE WE ARE HERE. The cache is sound only because the prefix is
    frozen, so this is where that is re-checked against the parameters themselves rather than
    against the flag string that was supposed to freeze them. A step boundary is immediately after
    an optimiser step, which is the only thing that could have moved them.
    """
    if not _S["on"]:
        return
    live = [n for n, p in _S["head"].named_parameters()
            if n in _S["prefix_names"] and p.requires_grad]
    if live:
        raise SystemExit(
            f"STOP: --prefix-cache-gb is on but {len(live)} prefix parameter(s) now require grad, "
            f"e.g. {live[:4]}. The cache holds an activation computed from those weights and hands "
            f"it to a later pass as a constant; if they are being trained, the cached value is "
            f"stale by one optimiser step and the gradient would be taken at the wrong point. "
            f"Either widen the cut or drop --prefix-cache-gb.")
    now = _digest(_S["prefix_params"])
    if now != _S["digest"]:
        raise SystemExit(
            f"STOP: --prefix-cache-gb is on and the FROZEN prefix changed value "
            f"({_S['digest']} -> {now}). Nothing in this arm should write those tensors, so "
            f"something -- an optimiser group, a weight load, an EMA -- is touching them, and "
            f"every cached activation taken before now describes a head that no longer exists.")
    with _LOCK:
        _S["gen"] += 1
        _S["cache"].clear()
        # THE RESERVATION IS NOT CLEARED, only the admitted bytes. A fill may be between its
        # reservation and its `finally`; clearing the dict underneath it would make that pop a
        # no-op and leave _resv permanently above zero. The generation check stops NEW entries,
        # which is what retiring a step means. Same reasoning as _tr_reset.
        _S["bytes"] = 0
    clear_pose()


def stats():
    """A copy of the counters, plus the current byte totals. For the epoch line."""
    with _LOCK:
        out = dict(_S["stat"])
        out["bytes"] = _S["bytes"]
        out["resv"] = _S["resv"]
        out["entries"] = len(_S["cache"])
        out["cap"] = _S["cap"]
    return out


def reset_stats():
    _S["stat"] = _zero_stats()


def install(model, cap_gb, *, lean_single, device=None, loud=True,
            crop_pool="", crop_all_poses=False):
    """Turn the cache on, after REFUSING everything that would make it unsound.

    Every precondition here is a hard stop, not a downgrade. A cache that silently declines to
    engage is indistinguishable from one that engages and is wrong, and the arm's epoch time is the
    only signal either way.
    """
    cap = int(float(cap_gb) * (1 << 30))
    if cap <= 0:
        _S["on"] = False
        return None
    # ---- REFUSE ANY CROP, which is the one failure mode that is silent in BOTH directions ------
    #
    # The key carries the token COUNT, so a unit re-entered at a different N is a miss. That is not
    # enough. A crop can hand pass A and pass B the SAME N over DIFFERENT kept residues -- same
    # shape, same key, different inputs -- and the cached z is then a valid-looking activation for
    # a structure that was not the one being differentiated. No shape error, no NaN, just wrong
    # gradients that look plausible.
    #
    # run_unit's pass-A and pass-B paths are not the same code: pass A crops inline while pass B
    # can consume a prefetched tensor, so "the crop is a pure function of the seed" is a property
    # of two code paths agreeing and not something this module can assert from here.
    #
    # The arm this was written for is FULLY UNCROPPED, so this costs nothing today. It is here
    # because the combination is unreasoned rather than known-bad, and the failure would be a
    # wrong number rather than a crash. Key on the crop's identity and lift this if cropping is
    # ever wanted with the cache.
    if crop_pool or crop_all_poses:
        raise SystemExit(
            "STOP: --prefix-cache-gb with cropping (--crop-pool "
            f"{crop_pool!r}, --crop-all-poses {crop_all_poses}). "
            "The cache key carries the token COUNT but not WHICH tokens were kept, so a crop that "
            "gives pass A and pass B the same N over different residues would be a cache HIT on "
            "an activation computed from different inputs -- wrong gradients, no error. This "
            "combination has not been reasoned through and is refused rather than ordered. The "
            "arm this was built for is fully uncropped.")
    head = getattr(model, "confidence_head", None)
    if head is None:
        raise SystemExit("STOP: --prefix-cache-gb: the model has no confidence_head.")

    # 1. THE INJECTION MUST BE INSTALLED, because the cut is a parameter of ITS mirror of
    #    PairformerBlock.forward and there is nowhere else to take it.
    if not getattr(head, "_pose_inblock_installed", False):
        raise SystemExit(
            "STOP: --prefix-cache-gb without --pose-inblock. The cut is INSIDE the last pairformer "
            "block, between tri_mul_in and the pose injection, and phases/pose_inblock.py is the "
            "only code that can execute a block in two halves. Without it the deepest frozen "
            "boundary is the stack's INPUT -- which is z_trunk plus the native injection, i.e. "
            "nothing worth caching -- and this flag would have no legal place to cut.")
    import pose_inblock as PIB

    # 2. --lean-single, because 'the suffix does not need s' is a consequence of it and of nothing
    #    else. CHECKED ON THE BLOCKS, not on the flag: c_s is what the forward branches on.
    blocks = list(head.pairformer_stack.blocks)
    bad_cs = [i for i, b in enumerate(blocks) if getattr(b, "c_s", 0) > 0]
    if bad_cs or not lean_single:
        raise SystemExit(
            f"STOP: --prefix-cache-gb requires --lean-single (flag={bool(lean_single)}, blocks "
            f"with c_s > 0: {bad_cs}). The cache stores `z` ALONE. That is only complete because "
            f"pose_inblock's mirror updates the single track under `if blk.c_s > 0:` and "
            f"--lean-single sets c_s = 0 on every block, so `s` crosses the stack untouched. With "
            f"the single track live, `s` at the cut is part of the state the suffix reads and "
            f"resuming from `z` alone would silently use the WRONG `s`.")

    # 3. THE BLOCK INDEX, read from what pose_inblock actually INSTALLED and not from the flag.
    #    --pose-inblock-block defaults to -1 and is resolved to the last block inside install(),
    #    so the flag's value and the installed depth are different numbers.
    block_index = int(getattr(head, "_pose_inblock_block_index", -1))
    if not 0 <= block_index < len(blocks):
        raise SystemExit(
            "STOP: --prefix-cache-gb cannot tell which block the pose was injected into. "
            "pose_inblock.install() must record it on the head as _pose_inblock_block_index; "
            "guessing the depth would put the cut in the wrong place, which is a different "
            "operator and nothing downstream could tell.")
    if getattr(head, "_pose_inblock_inject_at", None) != "after_tri_mul_in":
        raise SystemExit(
            f"STOP: --prefix-cache-gb with inject_at="
            f"{getattr(head, '_pose_inblock_inject_at', None)!r}. The cut is defined as 'after "
            f"tri_mul_in, before the injection'. At the block INPUT the pose -- and therefore a "
            f"trainable parameter -- is UPSTREAM of both tri-muls, so there is no frozen prefix "
            f"inside the block at all.")

    # 4. DETERMINISM. p_drop on every block the prefix runs through, read at runtime rather than
    #    taken from protenix's configs_base.py, and including the nn.Dropout the block holds --
    #    --pair-dropout sets p_drop directly on built blocks, so the config file is not the answer.
    drops = []
    for i in range(block_index + 1):
        b = blocks[i]
        drops.append(("p_drop", i, float(getattr(b, "p_drop", 0.0))))
        for n, m in b.named_modules():
            if isinstance(m, torch.nn.Dropout):
                drops.append((f"{n}.p", i, float(m.p)))
    live_drop = [d for d in drops if d[2] != 0.0]
    if live_drop:
        raise SystemExit(
            f"STOP: --prefix-cache-gb needs the prefix to be DETERMINISTIC and it is not: "
            f"{live_drop[:4]} (of {len(live_drop)} non-zero rate(s) across blocks "
            f"0..{block_index}). A cached prefix is one realisation of a stochastic function; "
            f"handing it to pass B makes pass B evaluate a different function than pass A scored, "
            f"and the two passes would disagree by the dropout draw with nothing reporting it. "
            f"Issue #8 step 3 is explicit that a stochastic prefix must not be cached.")

    # 5. THE PREFIX IS GENUINELY FROZEN, by requires_grad on the tensors, not by the --train-only
    #    string that was supposed to set it.
    names = _prefix_param_names(head, block_index)
    if not names:
        raise SystemExit(
            f"STOP: --prefix-cache-gb found NO prefix parameters at block_index={block_index}. "
            f"Either the module tree is not what this module was written against or the cut is at "
            f"the very front of the head, in which case there is nothing to cache and the flag is "
            f"a no-op that would cost a D2H per pose.")
    params = dict(head.named_parameters())
    live = [n for n in names if params[n].requires_grad]
    if live:
        raise SystemExit(
            f"STOP: --prefix-cache-gb: {len(live)} of {len(names)} prefix parameter(s) require "
            f"grad, e.g. {live[:6]}. The cache is sound ONLY because the prefix cannot move "
            f"between pass A and pass B. Under the default arm --train-only leaves block "
            f"{block_index}'s tri_att_start / tri_att_end / pair_transition, the pose_inblock_* "
            f"projections and the PAE readout trainable and nothing else; this set says otherwise, "
            f"so either --train-only is wider than that or the cut is in the wrong place.")

    _S.update({
        "on": True, "cap": cap, "bytes": 0, "resv": 0, "cache": {}, "inflight": {}, "gen": 0,
        "head": head, "block_index": block_index,
        "prefix_names": set(names),
        "prefix_params": [params[n] for n in names],
        "dev": torch.device(device) if device is not None else None,
        "c_z": int(head.c_z), "n_tok": -1,
        "key": None, "fill": False,
    })
    _S["digest"] = _digest(_S["prefix_params"])
    _S["stat"] = _zero_stats()

    # THE SIGNATURE. Every one of these changes what the cached tensor MEANS, so a change makes
    # every pre-existing entry unreadable rather than silently reusable. The pinned shas are in
    # because a Protenix upgrade that reorders the block moves the cut.
    sig_src = "|".join([
        SCHEMA,
        f"block={block_index}",
        f"inject={getattr(head, '_pose_inblock_inject_at', None)}",
        f"pae={bool(getattr(head, '_pose_inblock_hook', None))}",
        f"nblocks={len(blocks)}",
        f"c_z={int(head.c_z)}",
        f"bins={int(head.num_bins)}",
        f"lean_single=1",
        f"blocksha={PIB.PINNED_BLOCK_SHA}",
        f"prefix={','.join(sorted(names))}",
        f"digest={_S['digest']}",
    ])
    _S["sig"] = hashlib.sha256(sig_src.encode()).hexdigest()[:16]

    # 6. AND IT MUST ACTUALLY TAKE. A flag that prints a line and changes nothing is the failure
    #    mode --blocks-per-ckpt had for its whole life.
    PIB.CUT["plan"] = _plan
    if PIB.CUT["plan"] is not _plan:
        raise SystemExit("STOP: --prefix-cache-gb did not take: pose_inblock.CUT['plan'] is not "
                         "this module's planner after assignment.")

    n_pfx = sum(params[n].numel() for n in names)
    info = {"cap_bytes": cap, "block_index": block_index, "sig": _S["sig"],
            "prefix_tensors": len(names), "prefix_params": int(n_pfx)}
    if loud:
        print(f"PREFIX CACHE: {cap / (1 << 30):.2f} GiB per rank, cut INSIDE pairformer block "
              f"{block_index} after tri_mul_in and before the pose injection. The prefix is "
              f"{len(names)} frozen tensor(s) / {n_pfx:,} parameter(s): the native pose injection, "
              f"blocks 0..{block_index - 1} entirely, and block {block_index}'s tri_mul_out and "
              f"tri_mul_in. Verified at runtime: every one has requires_grad=False, every "
              f"dropout rate in blocks 0..{block_index} is 0.0, and --lean-single has c_s=0 on all "
              f"{len(blocks)} block(s) so `z` alone is the complete cut state. Scope is ONE "
              f"optimiser step: pass A fills, pass B hits, the step boundary drops it. Signature "
              f"{_S['sig']}. ADVISORY -- a refused or missing entry recomputes the prefix, so no "
              f"number the run produces depends on a hit.")
    return info


def uninstall():
    """Detach the planner and drop everything. For tests; the trainer never calls it."""
    try:
        import pose_inblock as PIB
        PIB.CUT["plan"] = None
    except Exception:                                    # noqa: BLE001
        pass
    with _LOCK:
        _S["cache"].clear()
        _S["inflight"].clear()
        _S["bytes"] = 0
        _S["resv"] = 0
        _S["on"] = False
        _S["key"] = None
        _S["fill"] = False
