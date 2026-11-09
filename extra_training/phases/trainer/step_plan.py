"""The epoch's step plan: which units each rank walks, in which optimiser step.

WHY THIS IS WORTH ITS OWN FILE. It answers one question -- "which poses belong to which step on
which rank?" -- and nothing in it touches the filesystem, torch or the network. It was 195 lines
inside an 8,380-line main(), reachable only by starting a training run, and it decides the thing
every later stage depends on: get the grouping wrong and the relative loss centres over the wrong
set of poses while every number downstream still looks plausible.

THE CONSTRAINT IT ENFORCES. --rank-cross-seed ranks poses across seeds, and one pose's coefficient
depends on every other pose in its interface, so the whole interface must be scored before any
backward in the step. Every seed of a structure therefore has to land in the SAME optimiser step,
which the normal epoch order deliberately does not do.

Dependencies are explicit parameters rather than closure reads; the body is otherwise verbatim.
phases/trainer/test_step_plan.py checks it against the unit order real runs actually walked.
"""
import random


def build_epoch_cross_seed(ep, a, flat, rank, world, lrs, lr_horizon, sel_tokens, xs_spe, q,
                           is_main=False):
    """(mine, quotas, n_units) for one epoch.

    ep          epoch index
    a           the parsed config
    flat        [(pdb, unit, ...)] every unit in the manifest
    rank/world  this process's rank and the world size
    lrs         the lr_schedule module
    lr_horizon  single-element list the caller uses to publish steps-per-epoch x epochs
    sel_tokens  {pdb: token count}, for the token-ordered modes
    xs_spe      single-element list for the cross-seed steps-per-epoch
    q           the quota accumulator the caller reads back

    mine    the flat unit list THIS rank walks, in order
    quotas  this rank's unit count per step -- the step boundary, since S structures x their
            seeds need not divide `world`
    n_units the GLOBAL unit count per step, which is what the PAE mean is taken over
    """
    # The body below is VERBATIM from phase_k_train.py. Its closure reads are bound to the
    # explicit parameters here rather than renamed throughout, so a diff against the original
    # shows no change to the logic itself.
    _LRS, _lr_horizon, _q = lrs, lr_horizon, q
    _sel_tokens, _xs_spe = sel_tokens, xs_spe

    """This epoch's unit sequence FOR THIS RANK, with ALL SEEDS OF A STRUCTURE IN ONE STEP.

    Returns (mine, quotas, n_units): the flat unit list, this rank's unit count per step, and
    the GLOBAL unit count per step. The step boundary comes from `quotas` rather than from a
    constant, because S structures x their seeds need not divide `world`; n_units is what the
    PAE term's mean has to be taken over, since accum_eff no longer describes the step.

    WHY THE REGROUPING. --rank-cross-seed ranks poses across seeds, and the coefficient for one
    pose depends on every other pose in its interface. So the whole interface has to be scored
    before any backward in the step, which means every seed of the structure must be inside the
    same optimiser step. build_epoch's normal order deliberately spreads a structure's ten
    seeds across ten different steps (one per rank per step), which is the opposite.

    THE STEP STAYS THE SAME SIZE. S=2 structures x 10 seeds x 5 poses = 100 poses, against the
    ~100 the normal schedule already uses (K=6 structures x 3 ranks x 5 poses). What changes is
    the SHAPE: 2 structures with all their seeds instead of ~6 structures with 3 seeds each. So
    the ranking signal becomes exact (the full ~1,225 cross-seed pairs of each interface rather
    than a 75-pair sample, measured cosine 1.000 vs 0.856) at the cost of each step's PAE
    gradient seeing a third as many structures.

    NO SEED SITS OUT. build_epoch chops seeds into whole groups of `world` and the remainder
    -- one seed per structure at world 3 -- does not train that epoch. Here a group is ALL the
    seeds, so every pose is used every epoch.

    THE SHORT RANK ROTATES. 20 units over 3 ranks is 7/7/6, and a rank that takes fewer units
    in every step would reach the end of the epoch having taken a different number of steps --
    which does not fail, it HANGS on the first mismatched all-reduce. So the extra units go to
    ranks (k + 0 .. k + rem - 1) mod world at step k, which gives every rank the same total
    whenever the step count is a multiple of `world`. When it is not, per-rank unit counts
    differ and the existing cross-rank count check refuses to start -- loudly, rather than
    deadlocking. UNITS ARE SHUFFLED WITHIN THE STEP so each rank gets a mix of the step's
    structures rather than one structure's seeds, which is what keeps per-rank token counts
    comparable; every seed of a structure crops to the same token count, so a balanced mix of
    structures is balanced work.
    """
    by = {}
    for _p, _lp, _c in flat:
        by.setdefault(_p, []).append((_lp, _c))
    rng_e = random.Random(a.seed + ep)
    # ASCENDING BY TOKENS when asked. The pdb tie-break keeps it deterministic across ranks
    # and epochs. A structure the manifest gave no numeric `tokens` for sorts first and is
    # named below rather than silently placed.
    _manifest_order = ('shuffle' in ("manifest", "tokens"))
    if 'shuffle' == "tokens":
        order = sorted(by, key=lambda _q: (_sel_tokens.get(_q, 0), _q))
    elif _manifest_order:
        order = list(by)
    else:
        order = sorted(by)
        rng_e.shuffle(order)
    S = max(1, int(a.rank_structs_per_step))
    # THE STEP IS A CONSTANT NUMBER OF UNITS, NOT A CONSTANT NUMBER OF STRUCTURES. With a
    # uniform manifest the two are the same thing and this packs S structures per step exactly
    # as before. With mixed depth they are not: 437 structures carry 15 seeds and 416 carry 1,
    # so S=2 structures per step would mean 30 units for one pair and 2 for another.
    #
    # THAT WOULD HAVE BEEN SILENT, which is why the packing changed rather than the quotas.
    # Three sites downstream read the step's size as a per-EPOCH constant:
    #   n_units = len(steps[0])        quotas for EVERY step come from the FIRST step's length
    #   _step_poses_g[0]               the divisor of the PAE term's per-step mean
    #   the _xs_steps walk            slices ep_units by those quotas and demands they sum
    # A longer step would be sliced to the first step's length and the rest of its units simply
    # never trained; a shorter one would hand a rank fewer units than its quota -- python slices
    # do not raise -- and the ranks would take different step counts and HANG on an all-reduce.
    # Keeping every step the same unit count leaves all three correct as written.
    #
    # THE BUDGET IS S x THE DEEPEST STRUCTURE, so a step is the same size in poses as it was.
    # Structures are taken in `order` and any whose depth does not fit the remaining budget is
    # passed over for a later one that does -- 15 ones fill a 15-unit remainder exactly, which
    # is why the two populations pack with nothing left over.
    # THE PACKING IS _LRS.pack_cross_seed_steps, WHICH IS ALSO WHAT SETS THE LR HORIZON.
    # It used to be an inline loop here and a formula there, and they drifted: the horizon
    # claimed len(order)//S = 426 steps while this built 232, a 1.84x error on the decay
    # window. One implementation cannot disagree with itself. The sort inside it is by
    # descending depth and STABLE, so shuffling `order` first keeps per-epoch randomness in
    # which structures share a step while making the COUNT a function of the depth histogram
    # alone -- plain first-fit over a shuffled order gave 220..229 steps on this population,
    # so no constant steps-per-epoch existed to calibrate against.
    _depth = {_p: len(by[_p]) for _p in order}
    _chunks, _dropped, U = _LRS.pack_cross_seed_steps(_depth, order, S,
                                                       preserve_order=_manifest_order)
    # THE DEEPEST STRUCTURE'S UNIT COUNT. Both diagnostics below interpolate `_maxd` and
    # nothing ever bound it: `NameError: name '_maxd' is not defined`, raised from the
    # _manifest_order branch, which the arm never takes and a size-ordered memory probe always
    # does. So the one configuration that exists to find the memory ceiling died before
    # reaching a single pose.
    #
    # Derived from _depth rather than from U, and then checked against U, so the message
    # cannot drift from the packing it describes -- lr_schedule.pack_cross_seed_steps:154
    # computes U = S * max(depth), and if that ever changes this stops rather than printing a
    # number that is no longer true.
    _maxd = max(_depth.values()) if _depth else 0
    if _chunks and U != S * _maxd:
        raise SystemExit(
            f"STOP: a cross-seed step is {U} unit(s) but S={S} times the deepest structure "
            f"({_maxd}) is {S * _maxd}. build_epoch_cross_seed's diagnostics describe a step "
            f"as 'S x deepest', and pack_cross_seed_steps no longer agrees.")
    steps = []
    for chunk in _chunks:
        # WHICH SEEDS, AND IN WHAT ORDER, STAYS RANDOM -- that is this shuffle, and it is the
        # one that makes a structure's 15 units arrive in a different order each epoch.
        _lists = []
        for _p in chunk:
            u = by[_p][:]
            if not _manifest_order:
                rng_e.shuffle(u)
            _lists.append([(_p, _lp, _c) for _lp, _c in u])
        if _manifest_order:
            units = [x for _l in _lists for x in _l]
        else:
            # DEAL THE STEP ROUND-ROBIN ACROSS ITS STRUCTURES, DO NOT SHUFFLE IT GLOBALLY.
            # The split below takes CONTIGUOUS slices, so the order of this list IS the rank
            # assignment. A global shuffle gave each rank a random 10 of the step's 30 units,
            # which for a step of 2 deep structures means a random split of 15+15 -- so one
            # rank could hold 8 units of the larger structure and another 2. Memory per unit
            # goes as tokens^2 (measured: peak_GB = 2.02 + 2.896e-05 * tokens^2, residual
            # under 0.3 GB from 268 to 1364 tokens), so an uneven split is an uneven WAIT:
            # every rank sits at the step's all-reduce until the slowest finishes.
            #
            # MEASURED ON THE 841-STRUCTURE POPULATION, per-step tokens^2 work, max-min over
            # min: global shuffle median 17.2% / mean 18.5% / p90 40.1%, 90 of 230 steps worse
            # than 20%. Round-robin: median 0.0% / mean 2.3% / p90 0.0%, 13 of 230. The 13 are
            # the all-shallow steps -- 30 distinct 1-seed structures, where no assignment can
            # give three ranks the same structures and the imbalance is irreducible here.
            #
            # IT IS EXACT WHENEVER EVERY STRUCTURE'S DEPTH DIVIDES `world`: 216 of the 230
            # steps are 2 structures x 15 seeds, 15 = 3 x 5, so each rank gets exactly 5 units
            # of each -- the balance is structural, not probabilistic.
            #
            # RANDOMNESS IS PRESERVED WHERE IT MATTERS: the rotation's starting structure is
            # shuffled, so it is not always the alphabetically-first one that leads, and the
            # per-structure seed order above is untouched. What is removed is only the random
            # ASSIGNMENT of units to ranks, which carried no training signal -- the ranking
            # coefficients are gathered across all ranks before any backward, so which rank
            # holds which seed cannot change the objective.
            rng_e.shuffle(_lists)
            units = _LRS.deal_round_robin(_lists)
        steps.append(units)
    # THE HORIZON AND THE BUILDER AGREE, ASSERTED EVERY EPOCH. The LR schedule was
    # calibrated once at startup from _LRS.steps_per_epoch; if this epoch packs a different
    # number of steps then lr_at_step is reading the wrong horizon and the arm is not running
    # the schedule it reports. Cheap, and it is the check whose absence let 426-vs-232 ship.
    if steps and _lr_horizon[0] and _xs_spe[0] and len(steps) != _xs_spe[0]:
        raise SystemExit(
            f"STOP: epoch {ep} packed {len(steps)} cross-seed step(s) but the lr horizon was "
            f"built from {_xs_spe[0]} step(s)/epoch. The decay window would be wrong by "
            f"{_xs_spe[0] / max(1, len(steps)):.3f}x. Both come from "
            f"_LRS.pack_cross_seed_steps, so this means the population changed mid-run.")
    if steps and is_main and len({len(x) for x in steps}) != 1:
        # NOT REACHABLE BY CONSTRUCTION, asserted anyway: everything downstream treats the
        # step size as a constant, and a packing bug here is the one that deadlocks instead of
        # failing.
        raise SystemExit(f"STOP: cross-seed steps are not all the same unit count: "
                         f"{sorted({len(x) for x in steps})}. The quota bookkeeping, the PAE "
                         f"per-step mean and the _xs_steps walk all assume one size.")
    if not steps:
        raise SystemExit(
            f"STOP: --rank-cross-seed with --rank-structs-per-step {S} needs at least {S} "
            f"structure(s); the manifest resolved {len(order)}.")
    if _manifest_order and is_main:
        # SAY WHAT WILL ACTUALLY RUN, AND WHAT WILL NOT. A short final step is dropped rather
        # than padded (see above), and in a size-sorted manifest the dropped chunk is the
        # LARGEST structures -- precisely the ones a memory probe exists to reach. A probe
        # that silently never ran them would recommend a threshold from the wrong structure.
        print(f"  EPOCH ORDER manifest: {len(order)} structure(s), "
              f"{len(steps)} step(s) of {U} unit(s) (S={S} x deepest {_maxd}); "
              f"first {order[:3]} last {order[-3:]}")
        if _dropped:
            print(f"  EPOCH ORDER manifest: {len(_dropped)} structure(s) DROPPED as a short "
                  f"final step: {_dropped}. A step is {U} unit(s) "
                  f"(--rank-structs-per-step {S} x deepest structure {_maxd}); these could not "
                  f"fill one exactly. Trim the FRONT of the manifest or adjust S, or these are "
                  f"never run.", flush=True)
    if _dropped and _manifest_order:
        # REFUSE RATHER THAN WARN. Raised in review: in a size-sorted order the dropped chunk
        # is the LARGEST structures, which is precisely what a memory probe exists to reach, so
        # a probe that silently never ran them would report a threshold from the wrong
        # structure. A warning in a log is not a gate. The shuffled modes keep the old
        # behaviour because there the omitted set rotates with the epoch seed.
        raise SystemExit(
            f"STOP: --epoch-order {'shuffle'} dropped {len(_dropped)} structure(s) as a "
            f"short final step: {_dropped[:6]}. In a deterministic order the dropped chunk is "
            f"the largest structures and they would never train, which silently changes what a "
            f"size-ordered epoch means. Trim the manifest to a multiple of the "
            f"{U}-unit budget, or adjust --rank-structs-per-step.")
    n_units = len(steps[0])
    base, rem = divmod(n_units, world)
    mine, quotas = [], []
    for k, units in enumerate(steps):
        counts = [base + (1 if ((r - k) % world) < rem else 0) for r in range(world)]
        start = sum(counts[:rank])
        mine += units[start:start + counts[rank]]
        quotas.append(counts[rank])
    return mine, quotas, n_units

