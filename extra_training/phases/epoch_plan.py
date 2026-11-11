"""The cover-all epoch plan, as a module-level pure function.

WHY IT LIVES HERE AND NOT AS A CLOSURE. External review of 216b0630 asked for a CPU simulation of
the REAL builder rather than a duplicate reimplementation, and the builder it was reviewing was a
closure inside phase_k_train.main() -- unreachable from a test without standing up the model. A
reimplementation in the test would assert that the copy is correct, which is the one thing nobody
needs to know. So the logic is here, phase_k_train calls it, and test_epoch_cover_all.py calls the
same function with the same manifest.

WHAT IT REPLACES. phase_k_train.build_epoch's unit of work is `world` SEEDS OF ONE STRUCTURE:

    for g in range(len(u) // world):

At len(u) == 1 that range is empty, so a single-seed structure produces no group and NEVER TRAINS,
in any epoch, with nothing in the log saying so. This population is 412 structures at 1 seed and
434 at 15, so 412 of 846 would sit out the whole arm. Same-structure groups exist to equalise
per-step work for the RANKING loss, which needs several poses of one structure in one step; a
PAE-only arm has no such requirement and a fixed-size crop equalises the work anyway.
"""


def cover_all(flat, world, rank, seed, ep, order="shuffle", tokens=None, balance=False,
              per_step=0):
    """This rank's units for epoch `ep`: every unit once, dealt round-robin across ranks.

    flat   [(pdb, label_path, crop), ...]   every unit of the epoch, in any order
    world  rank count;  rank  this rank's index;  seed  --seed;  ep  epoch index
    order  "shuffle" (default), "tokens" or "manifest", mirroring --epoch-order
    tokens {pdb: n_token} when order == "tokens"

    EQUAL COUNTS PER RANK, OR THE FIRST ALL-REDUCE HANGS instead of failing. len(flat) need not
    divide `world`, so the list is PADDED up to a multiple of it -- never truncated.

    PADDED, NOT TRUNCATED, AND THAT IS NOT A STYLE CHOICE. Truncating 6,922 units to 6,921 drops
    one unit per epoch, and the coverage gate caught it dropping 9zrm -- a SINGLE-SEED structure,
    whose only unit that was, so it missed the epoch entirely. That is a smaller instance of the
    exact defect this function exists to fix. Padding duplicates at most world-1 units instead,
    which costs one extra trunk read and weights one unit twice out of ~6,900.

    THE PADDING COMES FROM THE DEEPEST STRUCTURES, so the duplicate falls on a structure that
    already contributes 15 units rather than on a singleton, where it would be a 2x reweighting.

    units[rank::world] makes one step's units a contiguous global block split across the ranks,
    so a step of `structs_per_step` units per rank spans structs_per_step*world DISTINCT units.
    """
    import random

    units = list(flat)
    if order == "tokens":
        tokens = tokens or {}
        units.sort(key=lambda _t: (tokens.get(_t[0], 0), _t[0]))
    elif order != "manifest":
        random.Random(seed + ep).shuffle(units)
    # PAD FIRST, THEN DEAL. Every rank must walk the same NUMBER of units or the step
    # boundaries diverge and the first all-reduce hangs rather than failing.
    short = (-len(units)) % world
    if short:
        depth = {}
        for _p, _l, _c in flat:
            depth[_p] = depth.get(_p, 0) + 1
        pad = sorted(units, key=lambda _t: (-depth.get(_t[0], 0), _t[0]))[:short]
        units = units + pad

    if balance and tokens and per_step > 0:
        # BALANCE WITHIN EACH STEP, NOT ACROSS THE EPOCH.
        #
        # The per-EPOCH totals are already even -- measured 1.77% spread under plain dealing --
        # so balancing them changes nothing. The wait is per STEP: the ranks barrier at every
        # optimiser step, so each step costs its SLOWEST rank, and one rank drawing a 1,650-token
        # structure while another draws a 300-token one makes that step slow however even the
        # epoch totals are. Measured on the live arm, loss_stats_collective_plus_wait is 7.8 min
        # of a 19.9 min epoch (39%) while gloo's own all-reduce is documented at ~6 s per epoch,
        # so essentially all of it is wait.
        #
        # WHY SIZE AND NOT TOKEN COUNT: the crop equalises GPU work at 256 tokens, but a
        # 1,650-token structure still has a ~100 MB z_trunk to read before 97% of it is thrown
        # away. The cost that differs between ranks is the READ.
        #
        # LONGEST-PROCESSING-TIME-FIRST inside each block of per_step*world units, which is
        # exactly one optimiser step's worth. Deterministic, so every rank computes the same
        # assignment without communicating.
        blk = per_step * world
        out = []
        for b0 in range(0, len(units), blk):
            chunk = units[b0:b0 + blk]
            cap = (len(chunk) + world - 1) // world
            load = [0] * world
            cnt = [0] * world
            bins = [[] for _ in range(world)]
            for i in sorted(range(len(chunk)),
                            key=lambda j: (-tokens.get(chunk[j][0], 0), chunk[j][0])):
                r = min((x for x in range(world) if cnt[x] < cap),
                        key=lambda x: (load[x], x))
                bins[r].append(chunk[i])
                load[r] += tokens.get(chunk[i][0], 0)
                cnt[r] += 1
            out.extend(bins[rank])
        return out

    # ALREADY PADDED ABOVE, before the balance branch, so both paths deal the same list.
    return units[rank::world]


def coverage(flat, world, seed, ep, order="shuffle", tokens=None):
    """{pdb: how many of its units this epoch trains on}, summed over every rank."""
    seen = {}
    for r in range(world):
        for pdb, _lp, _c in cover_all(flat, world, r, seed, ep, order, tokens):
            seen[pdb] = seen.get(pdb, 0) + 1
    return seen
