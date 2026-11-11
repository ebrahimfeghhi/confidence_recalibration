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


def cover_all(flat, world, rank, seed, ep, order="shuffle", tokens=None):
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
    short = (-len(units)) % world
    if short:
        depth = {}
        for _p, _l, _c in flat:
            depth[_p] = depth.get(_p, 0) + 1
        # deepest first, then by pdb so two ranks computing this independently agree
        pad = sorted(units, key=lambda _t: (-depth.get(_t[0], 0), _t[0]))[:short]
        units = units + pad
    return units[rank::world]


def coverage(flat, world, seed, ep, order="shuffle", tokens=None):
    """{pdb: how many of its units this epoch trains on}, summed over every rank."""
    seen = {}
    for r in range(world):
        for pdb, _lp, _c in cover_all(flat, world, r, seed, ep, order, tokens):
            seen[pdb] = seen.get(pdb, 0) + 1
    return seen
