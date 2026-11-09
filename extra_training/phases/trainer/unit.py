"""`trainer/unit.py` -- score and backward one pose end to end.

EXTRACTED VERBATIM from `phases/phase_k_train.py` lines 4946-5666 (`run_unit`) (source `d34baebc`), dedented one
level from inside `main()`. Statements and comments are the original.

THESE WERE CLOSURES. Inside `main()` they read dozens of enclosing locals directly; here the
names they close over arrive as `ctx`, a plain namespace built by the caller. Nothing else about
them changed, and test_region_equivalence.py compares every line against the live source so a
change there fails the test instead of drifting.
"""
import numpy as np
import phase_j_rescore_cache as RC
# THE HELPERS main() DECLARED INLINE. Imported by name rather than with a star
# so the dependency is visible and a missing one is an ImportError here.
from helpers import (_gpu_span, _pose_pairs, _rank_key, _set_pae_mask, _tick, _tr_get, _tr_put, _unit_fp)
import csv
import math
import os
import time

import torch

def run_unit(rt, pdb, lab_p, trunk_p, score_only=False):
    # WHERE THE RESIDUAL IS, SPLIT ONCE. Epoch 17 left 14.9 min of 143.9 unaccounted -- the
    # second-largest single item in the table and the only one nobody can act on, because
    # "residual" does not say whether it is inside a unit or between units. unit_setup closes
    # the first half of that question: it covers everything from entering a unit to the moment
    # its first pose starts, which is the trunk read, the fingerprint, the dataset item and
    # the loss-feature prep that the existing per-step buckets step over. Whatever is left
    # after this is OUTSIDE the units, and that is a different fix.
    #
    # NOT A _gpu_span. This region launches no kernels of its own; a CUDA event here would
    # time the queue drain of the previous unit's poses and attribute it to setup.
    _t_unit_setup = time.perf_counter()
    """Forward + loss for every pose of one unit. Yields (pose_index, loss_tensor, metrics).

    score_only=True makes this PASS A: a no_grad forward per pose that yields
    {"pose": i, "iptm": {pair: value}} and nothing else -- no loss, no metrics, no backward,
    no graph. Needed only by --rank-form softplus, whose coefficients contain
    sigmoid(-m/tau) and therefore depend on the current scores. See rank_loss's
    THE MEMORY CONSTRAINT.

    Both paths are resolved by the manifest and asserted to exist before the run starts, so
    there is nothing here to skip.
    """
    arm_seed = os.path.basename(lab_p)[:-4]
    # THE UNIT'S SEED AS AN INT, built through _rank_key so there is ONE place that parses a
    # seed out of an arm name. The pair argument is unused by the seed field and is only there
    # because _rank_key returns the whole key.
    _xs_sd = _rank_key(pdb, arm_seed, (0, 0))[1] if rt.a.rank_cross_seed else None
    # THE FULL DATASET ITEM IS NEVER BUILT, AND NOW CANNOT BE. The loss reads nine feature
    # keys and two label keys and all eleven live in the cached label npz -- 0.11 MiB against
    # the item's ~518 MB, bit-identical on 50 of 50 comparisons. It was already dead code
    # under --minimal-items; the K-structures-per-step order makes it unrunnable besides,
    # because every rank now touches EVERY structure each epoch, so an item cache would need
    # all 488 resident per rank where 164 already breached the box's free-RAM floor.
    _t_lab = time.perf_counter()
    lab = np.load(lab_p, allow_pickle=True)
    _tick("label_open", _t_lab)
    # THE CACHE'S PROVENANCE IS REQUIRED, NOT OPTIONAL. build_cache stores a `fingerprint` of
    # the job cif, bioassembly, index rows, trunk and code the label was built from, plus a
    # `schema`. Training read neither, so a label written by older code or from inputs that have
    # since changed would be trained on with nothing noticing. The lf_* loss features are the
    # same story: `if kk in lab` silently fell back to the LIVE features, which are not the ones
    # the label was built against -- exactly the substitution the replay gate exists to detect.
    for req in ("fingerprint", "schema", "label_coordinate", "label_coordinate_mask",
                "sample_index"):
        if req not in lab:
            raise SystemExit(f"STOP: {lab_p} has no `{req}`. It was written by an older "
                             f"build_cache; rebuild it before training.")
    if int(np.asarray(lab["schema"])) != int(rt.BC.SCHEMA):
        raise SystemExit(f"STOP: {lab_p} is schema {int(np.asarray(lab['schema']))}, this code "
                         f"expects {int(rt.BC.SCHEMA)}. Rebuild the cache.")
    # THE FINGERPRINT IS COMPARED, NOT MERELY PRESENT. resolve_trunk already proved one trunk
    # reproduces it; this re-asserts against the trunk actually being used, so the two cannot
    # come apart if resolve_trunk is ever bypassed.
    _want = _unit_fp(pdb, trunk_p)
    _got = str(np.asarray(lab["fingerprint"]))
    if _got != _want:
        _dw = dict(x.split("=", 1) for x in _want.split("|") if "=" in x)
        _dg = dict(x.split("=", 1) for x in _got.split("|") if "=" in x)
        _df = sorted(k for k in set(_dw) | set(_dg) if _dw.get(k) != _dg.get(k))
        if not (_df == ["code"] and rt.a.allow_code_drift):
            raise SystemExit(f"STOP: {lab_p} fingerprint differs in {_df}. Rebuild it.")
    _missing_lf = [k for k in rt.BC.LOSS_FEATS if f"lf_{k}" not in lab]
    if _missing_lf:
        raise SystemExit(f"STOP: {lab_p} stores no lf_ array for {_missing_lf}. Falling back to "
                         f"the live features would score the label against features it was not "
                         f"built with. Rebuild the cache.")
    # BOTH LOADERS, as the replay does. load_trunk gives the NUMPY features that
    # check_alignment compares against the dataset's -- it calls np.asarray, so handing it the
    # device tensors raises "can't convert cuda:0 device type tensor to numpy". load_unit gives
    # the device tensors the head consumes, and reverses the bf16-as-uint16 storage.
    # check_alignment ran once per structure in the preflight --minimal-items requires.
    # THE CROP, WHEN ASKED FOR. The head's tensors and the loss's features come out of
    # slice_cached_unit together, from one pair of masks, so there is no path on which one side
    # is cropped and the other is not -- the misalignment the re-indexing gate exists to catch.
    # interface_crop_tokens returns (None, None) when a crop is not definable (fewer than two
    # chains with usable coordinates, or no cross-chain contact), and this trains on the whole
    # structure then, counted, rather than guessing at a window.
    crop_slice = None
    if crop_slice is not None:
        (feat_head, s_inputs, s_trunk, zt, poses, _nc,
         crop_feat_loss, crop_labels, _nb) = crop_slice
    else:
        crop_feat_loss = crop_labels = None
        # TIMED, AT LAST. This is the UNCROPPED path's only trunk read and it carried no timer
        # of any kind, which is why 31.0 of epoch 3's 35.9 min residual could be placed INSIDE
        # units and no further. load_unit now ticks `trunk_npz_open`, `trunk_npz_read` and
        # `trunk_h2d` through this closure's own accumulator, so the read and the device copy
        # land in the epoch table separately -- the first answers whether prefetching it would
        # pay (item 4 of docs/training_speedups.md), the second whether pinned staging would.
        # ~243 MB per unit and 4,600 units per rank per epoch, i.e. ~1.1 TB each way.
        #
        # THE STEP'S CACHED HOST ARRAYS, IF THEY ARE THERE. A hit skips the npz entirely and
        # leaves only the device copy; a miss reads inline AND OFFERS THE RESULT, so pass B
        # hits on whatever pass A had to read itself. Both are lookups -- see the cache's own
        # comment for why this is a dict and not the cropped path's pop queue.
        _host = _tr_get(trunk_p)
        if rt._tr_on and _host is None:
            _host = RC.read_unit_host(trunk_p, tick=_tick, memmap=rt.a.trunk_memmap)
            _tr_put(trunk_p, _host)
        feat_head, s_inputs, s_trunk, zt, poses, _nc = RC.load_unit(
            trunk_p, rt.dev, tick=_tick, memmap=rt.a.trunk_memmap, host=_host)

    _t_fc = time.perf_counter()

    def to_dev(d):
        return {k: (v.to(rt.dev) if torch.is_tensor(v) else v) for k, v in d.items()}
    if crop_feat_loss is not None:
        # THE CROPPED LOSS FEATURES, from the same masks as the head's tensors. These are the
        # per-atom and per-token arrays the loss indexes with -- plddt_m_rep_atom_mask,
        # pae_rep_atom_mask, frame_atom_index, has_frame and the rest -- so a full-size copy
        # here against a cropped prediction is the silent misalignment, not a shape error.
        feat_cache = crop_feat_loss
    else:
        # ONE SHARED BUILDER, so this and the equivalence gate cannot drift apart -- two
        # readers of one record diverging is how the reporting bug shipped twice.
        feat_cache, _ = rt.BC.minimal_loss_inputs(lab, 0, rt.dev, torch)

    n_pose = poses.shape[0] if hasattr(poses, "shape") else len(poses)
    # FROM THE HEAD'S FEATURES, which carry asym_id on both paths. feat_cache does not, on the
    # minimal path: the loss never reads it, so it is not in the nine.
    n_tok = int(feat_head["asym_id"].shape[-1]) if "asym_id" in feat_head else -1
    # SIZE-CONDITIONAL CHECKPOINTING, decided here because this is the first point in the unit
    # where the token count is known and no pose has run yet. Assigned on EVERY unit rather
    # than only when it changes, so the stack can never carry a previous unit's setting into
    # this one -- a silent carry-over would be an OOM on the next large unit.
    if rt.a.ckpt_token_threshold > 0:
        _bpc = (None if 0 <= n_tok <= rt.a.ckpt_token_threshold
                else (rt.a.blocks_per_ckpt or None))
        rt.model.confidence_head.pairformer_stack.blocks_per_ckpt = _bpc
        rt._ckpt_hist[0 if _bpc is None else 1] += 1
    # ONCE PER UNIT, BEFORE ANY POSE RUNS. From feat_cache, not feat_head: the trunk cache
    # carries has_frame but neither frame_atom_index nor pae_rep_atom_mask, and all three are
    # in LOSS_FEATS. Under cropping feat_cache IS crop_feat_loss, so these are the cropped
    # copies whose atom indices match the cropped coordinates the head is given.
    if rt.a.frame_direction:
        import pose_direction
        pose_direction.set_unit(feat_cache["frame_atom_index"], feat_cache["has_frame"],
                                feat_cache["pae_rep_atom_mask"], pdb=pdb)
    if rt.world > 1 and n_pose != rt.a.nsample:
        raise SystemExit(f"STOP: {pdb}/{arm_seed} holds {n_pose} pose(s), not --nsample "
                         f"{rt.a.nsample}. Across ranks that shifts the optimiser step onto a "
                         f"different pose and the next all-reduce deadlocks.")
    # ONE POSE, IN A FUNCTION THAT RETURNS NUMBERS.
    #
    # This was inline in the generator, which held each pose's logits, prediction dict and labels
    # in its frame ACROSS the yield -- so the caller ran backward and then began the next forward
    # while the previous pose's pair tensors were still referenced. Two fp32 pair-logit tensors at
    # 3,400 tokens x 64 bins are about 5.5 GiB, so the peaks overlapped for no reason. Review round
    # 4 reproduced the object-lifetime behaviour and was explicit that this is OVERLAPPING PEAKS,
    # not unbounded accumulation -- correcting my own claim of a leak, which rested on dividing
    # 90.4 GB by 720 poses and finding a number that happened to match a tensor size.
    #
    # A called function's frame is destroyed on return, so every large tensor here is freed before
    # the next pose allocates. backward() happens inside and only Python floats come out; the
    # optimiser step stays with the caller, which owns the accumulation count.
    #
    # EVERYTHING FROM to_dev's DEFINITION TO HERE is per-unit loss-feature preparation, which
    # sat entirely in the unaccounted residual. Review was right that BC.minimal_loss_inputs
    # was the wrong suspect for it: with every one of the 4,860 units cropping successfully,
    # crop_feat_loss is always set and both the per-unit and per-pose minimal_loss_inputs
    # branches are dead code for this workload.
    _tick("loss_feature_prep", _t_fc)

    _tick("unit_setup", _t_unit_setup)

    def _one_pose(i):
        x = poses[i:i + 1].to(rt.dev).float() if torch.is_tensor(poses) \
            else torch.as_tensor(poses[i:i + 1], dtype=torch.float32, device=rt.dev)
        _fe0, _fe1 = _gpu_span("gpu_forward")
        # WHICH POSE THE FROZEN-PREFIX CACHE IS LOOKING AT, and whether it may fill.
        #
        # (pdb, arm_seed, pose) IS ENOUGH OF AN IDENTITY because the cache is dropped at every
        # step boundary and a step holds each pose exactly once; n_tok goes in too, so a unit
        # re-entered at a different token count is a MISS rather than a shape that lines up.
        # fill=score_only is the pass: pass A computes the prefix and offers it, pass B takes
        # it. A fill in pass B would pay a D2H for a tensor nothing reads again.
        #
        # CLEARED IN A `finally`, so a forward that happens anywhere else in this process
        # cannot be keyed as this pose and hit on it. The plan is decided once, inside
        # _prep_blocks, so clearing it here does not disturb the backward that re-runs the
        # block under blocks_per_ckpt=1.
        if rt._pfc is not None:
            rt.PFC.set_pose((pdb, arm_seed, int(i)), fill=bool(score_only), n_tok=n_tok)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                plddt, pae, pde, resolved = rt.model.run_confidence_head(
                    input_feature_dict=feat_head, s_inputs=s_inputs, s_trunk=s_trunk,
                    z_trunk=zt, pair_mask=None, x_pred_coords=x,
                    triangle_multiplicative=rt.configs.triangle_multiplicative,
                    triangle_attention=rt.configs.triangle_attention, inplace_safe=False,
                    # CHUNKING WAS TRIED AND REMOVED: --chunk-size 64 cost 5.8x in speed for
                    # no measurable memory, and Protenix's own config says chunk_size "should
                    # set to null for normal training".
                    chunk_size=None)
        finally:
            if rt._pfc is not None:
                rt.PFC.clear_pose()
        if _fe1 is not None:
            _fe1.record()
        # WHERE THE HEAD'S OUTPUTS LANDED. Compared BY INDEX, because torch.device("cuda") is not
        # equal to torch.device("cuda:0") and the first version of this check fired on every pose
        # of a single-GPU run -- it killed the memory probe before it printed a single number.
        _di = rt.dev.index if rt.dev.index is not None else torch.cuda.current_device()
        _off = [n for n, t in (("plddt", plddt), ("pae", pae), ("pde", pde), ("resolved", resolved))
                if t.device.type != "cuda" or (t.device.index or 0) != _di]
        if _off:
            raise SystemExit(
                f"STOP: {pdb}/{arm_seed} pose {i}: the confidence head returned {_off} off "
                f"cuda:{_di}. Protenix offloads pae/pde to the CPU above "
                f"PROTENIX_CONF_OFFLOAD_TOKENS (default 2000) when the module is NOT in training "
                f"mode; confidence_head.training is {rt.model.confidence_head.training}.")
        pred = {"plddt": plddt.float(), "pae": pae.float(), "pde": pde.float(),
                "resolved": resolved.float(), "coordinate_mini": x}
        # EVERY NUMERIC ARRAY GOES TO THE DEVICE, not only those already tensors. A numpy array
        # left as numpy is converted by Protenix with torch.as_tensor, giving a CPU tensor, and
        # PDELoss then builds its bin boundaries on the CPU too, so the whole loss computes there.
        if crop_feat_loss is not None:
            # SLICED ONCE PER UNIT: the label coordinates for every pose were cropped together,
            # so per pose this only picks a row.
            _cc, _cm = crop_labels
            ld = {"coordinate": torch.as_tensor(_cc[i], dtype=torch.float32, device=rt.dev),
                  "coordinate_mask": torch.as_tensor(_cm[i].astype(np.float32), device=rt.dev)}
        else:
            _, ld = rt.BC.minimal_loss_inputs(lab, i, rt.dev, torch)
        # THE POSE'S SAMPLE INDEX, resolved ONCE here and reused by the mask, pass A and
        # pass B. Three readers previously resolved it separately, and the first version of
        # the rel wiring had one of them use the pose index instead.
        _samp_i = None
        if rt.a.rank_form == "rel" and rt.a.rank_loss:
            _sv_i = lab.get("sample_index") if hasattr(lab, "get") else None
            if _sv_i is None:
                try:
                    _sv_i = lab["sample_index"]
                except Exception:                 # noqa: BLE001
                    _sv_i = None
            if _sv_i is None:
                raise SystemExit(
                    f"STOP: --rank-form rel needs `sample_index` in {lab_p} to resolve this "
                    f"pose's chain pair; without it the PAE mask and the rel target could "
                    f"describe different interfaces.")
            _samp_i = int(np.asarray(_sv_i).reshape(-1)[i])
        # PAE IS MASKED TO THE SAME PAIRS rel RANKS, per pose, under --rank-form rel.
        _po = None
        if _samp_i is not None:
            _po = {(min(_x, _y), max(_x, _y))
                   for _ifc, _x, _y in _pose_pairs(pdb, lab_p, _samp_i)}
        _set_pae_mask(pdb, feat_head["asym_id"], feat_cache["has_frame"], arm_seed,
                      pairs_override=_po)
        # PASS A STOPS HERE. The scoring sweep needs the pose's ipTM and nothing else: no loss,
        # no metrics, no backward. Returning before calculate_losses is not an optimisation, it
        # is the point -- pass A must not touch the loss state that pass B reports, and it runs
        # inside torch.no_grad() so no graph is built for the forward it just did.
        #
        # THE SAME ipTM FUNCTION AS PASS B, deliberately. If the two disagreed, the coefficient
        # would have been computed for a different function than the one being differentiated.
        if score_only:
            import rank_loss as RKL
            # THE POSE INDEX IS NOT THE SAMPLE INDEX, and the external labels are keyed by
            # the SAMPLE. The label npz carries the permutation in `sample_index`; measured on
            # the TRAINING fold it is non-identity in 3971 of 4000 files (99.3%) over 120
            # distinct patterns, e.g. 10fj/free_500000 -> [0, 4, 3, 2, 1]. This file already
            # documents the same trap for --emit-iptm-rows, where writing `i` moved the
            # aggregate by only 0.0009 while 73.8% of rows carried another sample's DockQ.
            # The rel target repeated it until external review caught it: pose 1 was being
            # regressed onto pose 4's true iTM.
            # ONE RESOLUTION, SHARED. _samp_i was resolved above for the PAE mask; under rel
            # it is always set, and for the other forms pass A still needs it for its key.
            if _samp_i is not None:
                _samp_a = _samp_i
            else:
                _si_a = lab.get("sample_index") if hasattr(lab, "get") else None
                if _si_a is None:
                    try:
                        _si_a = lab["sample_index"]
                    except Exception:            # noqa: BLE001
                        _si_a = None
                if _si_a is None:
                    raise SystemExit(
                        f"STOP: pass A needs `sample_index` in {lab_p} to key the group's "
                        f"labels the way results.csv and the rel label cache do. Without it a "
                        f"pose would be trained against another pose's label.")
                _si_a = np.asarray(_si_a).reshape(-1)
                if i >= _si_a.shape[0]:
                    raise SystemExit(f"STOP: pose {i} has no sample_index entry in {lab_p} "
                                     f"({_si_a.shape[0]} entries).")
                _samp_a = int(_si_a[i])
            _sc = {}
            # KEYED BY WHAT IS STABLE ACROSS POSES. For rel the unit of comparison is the
            # reference interface; its asym pair is resolved PER POSE, so the key is the
            # iface and the pair is a per-pose detail of how the score is computed.
            if rt.a.rank_form == "rel":
                _items = [(_ifc, (_x, _y)) for _ifc, _x, _y
                          in _pose_pairs(pdb, lab_p, _samp_a)]
            else:
                _items = [(_p, _p) for _p in sorted(rt._scored_asym.get(pdb) or ())]
            for _k_, _p in _items:
                # POOLED THE SAME WAY AS PASS B. If these differed, the coefficients would
                # describe one quantity and the backward would move another.
                _v = RKL.iptm_pooled(pred["pae"], feat_head, rt.configs, _p,
                                     pool=rt.a.rank_pool)
                if _v is not None:
                    _sc[_k_] = float(_v.detach())
            del pred, x, plddt, pae, pde, resolved
            return {"pose": i, "sample": _samp_a, "iptm": _sc}
        if rt.a.emit_iptm_rows and rt._scored_asym.get(pdb):
            # THE SAME FUNCTION PASS A USES. If this computed ipTM differently from the
            # ranking loss's own scorer, the emitted rows would describe a statistic the run
            # does not otherwise use, which is the failure the pass A comment warns about.
            import rank_loss as RKL
            if rt._iptm_fh["fh"] is None:
                rt._iptm_fh["fh"] = open(rt.a.emit_iptm_rows, "w", newline="")
                rt._iptm_fh["w"] = csv.writer(rt._iptm_fh["fh"])
                rt._iptm_fh["w"].writerow(["iface", "seed", "sample", "iptm"])
            _unit = os.path.basename(lab_p).rsplit(".", 1)[0]
            # THREE ARGUMENTS. _rank_key(pdb, arm_seed_or_int, pair) returns a key whose
            # second element is the parsed seed; calling it with one argument raised
            # TypeError on the FIRST emitted row, so --emit-iptm-rows could never have
            # produced a single line. Every other call site in this file passes three and
            # takes [1] for the seed -- see the pass A key at the --rank-cross-seed site.
            # Found in review of 511519e7.
            _seed = _rank_key(pdb, _unit, (0, 0))[1]
            # THE POSE INDEX IS NOT THE SAMPLE INDEX. The label npz carries an explicit
            # pose -> sample permutation in `sample_index`, written by
            # phase_j_build_cache.sample_indices from COORDINATE IDENTITY against the
            # <pdb>_sample_<k>.cif files, and it is almost never the identity: measured
            # non-identity on 396 of 400 validation label files. phase_k_eval writes the
            # MAPPED k, and results.csv is keyed by it. Writing `i` here looked right -- the
            # aggregate moved by 0.0009 -- while 73.8% of rows carried another sample's DockQ
            # and 56 of 92 interfaces selected a different pose. Caught in review.
            _si = lab.get("sample_index") if hasattr(lab, "get") else None
            if _si is None:
                try:
                    _si = lab["sample_index"]
                except Exception:            # noqa: BLE001
                    _si = None
            if _si is None:
                raise SystemExit(
                    f"STOP: --emit-iptm-rows needs `sample_index` in {lab_p} to key rows the "
                    f"way results.csv does, and this label does not carry it. Without it the "
                    f"rows would join to another sample's DockQ.")
            _si = np.asarray(_si).reshape(-1)
            if i >= _si.shape[0]:
                raise SystemExit(f"STOP: pose {i} has no sample_index entry in {lab_p} "
                                 f"({_si.shape[0]} entries).")
            _samp = int(_si[i])
            for _p in sorted(rt._scored_asym.get(pdb) or ()):
                # POOLED THE SAME WAY AS PASS B. If these differed, the coefficients would
                # describe one quantity and the backward would move another.
                _v = RKL.iptm_pooled(pred["pae"], feat_head, rt.configs, _p,
                                     pool=rt.a.rank_pool)
                if _v is None:
                    continue
                _ifc = rt._iface_of.get((pdb, _p))
                if _ifc is None:
                    continue
                # FULL PRECISION, NOT SIX DECIMALS. dockq_from_iptm.py already records that
                # the emitter's 6 dp makes ties MORE likely than the eval's and that 1 of 92
                # interfaces on j583i8 epoch 1 tied at that precision with two different
                # DockQ values behind it -- so rounding here can change the SELECTED POSE and
                # therefore the metric. repr() round-trips exactly. Found in review.
                rt._iptm_fh["w"].writerow([_ifc, _seed, _samp, repr(float(_v.detach()))])
                rt._iptm_fh["n"] += 1
            rt._iptm_fh["fh"].flush()
        try:
            cum, met = rt.loss_fn.calculate_losses(feat_cache, pred, ld, mode="train")
        except RuntimeError as _e:
            if "same device" not in str(_e):
                raise
            def _where(d, label):
                out = []
                for _k2, _v2 in sorted(d.items()):
                    if torch.is_tensor(_v2):
                        out.append(f"{_k2}:{_v2.device}")
                    elif isinstance(_v2, np.ndarray):
                        out.append(f"{_k2}:numpy[{_v2.dtype}]")
                return f"  {label}: " + " ".join(out)
            print(f"DEVICE MISMATCH on {pdb}/{arm_seed} pose {i}, dev={rt.dev}: {_e}", flush=True)
            print(_where(ld, "label_dict"), flush=True)
            print(_where(feat_cache, "feat"), flush=True)
            print(_where(pred, "pred"), flush=True)
            raise
        absent = [k for k in rt.TERMS if k not in met and k not in rt._dropped_terms]
        if absent:
            raise SystemExit(f"STOP: the loss produced no {absent} for {pdb}/{arm_seed} pose {i}; "
                             f"a term that stopped being produced reads as a lower loss, which is "
                             f"the failure this check exists for.")
        if not math.isfinite(float(met["loss"])):
            raise SystemExit(f"STOP: {pdb}/{arm_seed} pose {i} loss is {float(met['loss'])}, not "
                             f"finite. Training on it would destroy the head.")
        out = {k: (float(met[k]) if k in met else 0.0) for k in rt.TERMS}
        out["loss"] = float(met["loss"])
        out["pose"] = i
        if float(0.0) > 0:
            out["pae_loss_unweighted"] = rt._pae_unw["v"]
        # SCALED BY THE EFFECTIVE BATCH, so summing the ranks reconstructs the full-batch mean.
        # NOT UNDER --loss-only: there are no gradients to accumulate, and calling backward
        # would be the one way this mode could change the weights it is supposed to measure.
        # TWO SOURCES OF c_i, AND ONLY ONE IS ACTIVE. Per-unit (_rank_c, from the labels
        # alone) for the within-seed linear form; per-step (_xs_coef, from pass A) for
        # --rank-cross-seed, where a pose's coefficient depends on every other pose of its
        # interface and the divisor is interface-groups per step rather than units.
        if rt.a.rank_cross_seed:
            _cs_pairs = sorted({_k[3] for _k in rt._xs_coef
                                if _k[0] == pdb and _k[1] == _xs_sd and _k[2] == i})
            _cs_src = [(_p, rt._xs_coef.get((pdb, _xs_sd, i, _p))) for _p in _cs_pairs]
            _cs_div = float(rt._xs_div[0])
        elif _rank_on:
            _cs_src = [(_p, _rank_c.get((i, _p)))
                       for _p in sorted({_k[1] for _k in _rank_c if _k[0] == i})]
            _cs_div = float(rt._rank_div)
        else:
            _cs_src = []
            _cs_div = 1.0
        # BOUND BEFORE THE BACKWARD BLOCK, not inside it. The first version computed this
        # below the --grad-probe block that reads it, so the probe raised NameError on its
        # first pose -- the same defect review found at line 1067 with `want`. One binding,
        # above both readers.
        if not rt.a.loss_only and not score_only:
            _be0, _be1 = _gpu_span("gpu_backward")
            # UNDER --grad-probe: the two terms are measured with torch.autograd.grad,
            # which returns gradients WITHOUT touching any parameter's .grad. The first
            # version used .backward() with opt.zero_grad() around it, which wiped every
            # earlier pose's contribution to the open accumulation window -- so the claim that
            # the probe took the same step as a real run was false, and the step it did take
            # was built from one pose. Found in review of a3086ef.
            #
            # ACCUMULATED, NOT AVERAGED. The optimiser sees sum_i g_i, and ||sum_i g_i|| is
            # not sum_i ||g_i||: the ranking coefficients sum to zero inside a group, so
            # per-pose contributions can cancel heavily and a mean of per-pose norms would
            # overstate the ranking gradient at the step. Both are reported, and alpha is read
            # off the accumulated pair.
            # THE PROBE READS THE SAME COEFFICIENT SOURCE THE OBJECTIVE DOES. _rank_on is set
            # only on the per-unit linear path, so gating on it alone made the probe silently
            # measure nothing under --rank-cross-seed -- it would have reported a ranking
            # gradient of zero and the parity alpha as infinite.
            # ONE RESOLUTION, ABOVE BOTH READERS. The first version resolved the per-pose
            # chain pair only inside the backward block, so the --grad-probe block below --
            # which has its own scoring loop -- passed the IFACE NAME straight to
            # iptm_pooled and died on `i, j = pair` with "too many values to unpack". Found by
            # actually running the probe; the static gate had passed. This file has the same
            # lesson written at line 3707 about `_cs_div`: one binding, above both readers.
            _si_b = _samp_i
            # NAMED _rel_pp, NOT _pp. The --grad-probe block below iterates
            # `for (_n, _pp), _g in zip(trainable_params, _gs)`, so a dict called _pp in this
            # scope is shadowed by a torch Parameter by the time _asym_for reads it --
            # "AttributeError: 'Parameter' object has no attribute 'get'". Found by running
            # it; the static gate could not see a name collision.
            _rel_pp = None
            if rt.a.rank_form == "rel" and _cs_src:
                if _si_b is None:
                    raise SystemExit(
                        f"STOP: pass B has no sample index for {pdb} pose {i}; it is resolved "
                        f"once beside the PAE mask and must be set under --rank-form rel.")
                _rel_pp = {_ifc: (_x, _y) for _ifc, _x, _y
                           in _pose_pairs(pdb, lab_p, _si_b)}

            def _asym_for(_k):
                """The asym pair to score for a coefficient key.

                Under rel the key is a reference interface name and the pair is per pose;
                under every other form the key IS the pair.
                """
                if _rel_pp is None:
                    return _k
                _v = _rel_pp.get(_k)
                if _v is None:
                    raise SystemExit(
                        f"STOP: no per-pose chain pair for interface {_k} on {pdb} pose {i}; "
                        f"its coefficient cannot be applied to the score it describes.")
                return _v

            if rt.a.grad_probe and _cs_src:
                _rt = None
                for _p, _cv in _cs_src:
                    if not _cv:
                        continue
                    # SAME POOLING AS THE BACKWARD. This read RKL_probe.iptm, which is
                    # MAX-pooled, while training's backward uses iptm_pooled(pool=rank_pool).
                    # So the first rel probe's headline numbers -- parity alpha 121.7 and
                    # cos(PAE,rel) +0.2043 -- described the max-pooled score and NOT the
                    # objective being trained. Caught in external review; the parity must be
                    # re-measured.
                    _sp_ = rt.RKL_probe.iptm_pooled(pred["pae"], feat_head, rt.configs,
                                                 _asym_for(_p), pool=rt.a.rank_pool)
                    if _sp_ is None:
                        continue
                    _tm = (rt.a.alpha_rank * _cv / _cs_div) * _sp_
                    _rt = _tm if _rt is None else _rt + _tm
                # THE COEFFICIENT THIS POSE WAS PUSHED WITH, recorded alongside the
                # gradient it produced, so the two factors of
                #     g_i = c_i * grad_theta s_i
                # are measured ON THE SAME POSE IN THE SAME WINDOW. Without this the only
                # coefficient figures available were val10 sum|c_i| against a training-set
                # gradient probe, which cannot be divided -- raised in review, and correct.
                # Summed over the pose's interfaces exactly as the term above is built, with
                # the same alpha_rank and _cs_div scaling, so |g_rank| / this quantity is the
                # realised sensitivity |grad_theta s_i|.
                _cabs = sum(abs(rt.a.alpha_rank * _cv / _cs_div)
                            for _p, _cv in _cs_src if _cv)
                _pr_tensors = [pp for _n, pp in rt.trainable_params]
                _pr_norms = {}
                for _nm, _t in (("pae", cum / rt._step_poses_g[0]), ("rank", _rt)):
                    if _t is None:
                        continue
                    _gs = torch.autograd.grad(_t, _pr_tensors, retain_graph=True,
                                              allow_unused=True)
                    _flat = []
                    for (_n, _pp), _g in zip(rt.trainable_params, _gs):
                        _gv = torch.zeros_like(_pp) if _g is None else _g.detach().float()
                        rt._probe["sum"].setdefault(_nm, {})
                        _prev = rt._probe["sum"][_nm].get(_n)
                        rt._probe["sum"][_nm][_n] = _gv.clone() if _prev is None else _prev + _gv
                        rt._probe["finite"][_nm] = (rt._probe["finite"].get(_nm, True)
                                                 and bool(torch.isfinite(_gv).all()))
                        rt._probe["nonzero"].setdefault(_nm, set())
                        if float(_gv.abs().sum()) > 0:
                            rt._probe["nonzero"][_nm].add(_n)
                        _flat.append(_gv.flatten())
                    _pr_norms[_nm] = torch.cat(_flat)
                if "pae" in _pr_norms and "rank" in _pr_norms:
                    _vp, _vr = _pr_norms["pae"], _pr_norms["rank"]
                    _c = (float((_vp @ _vr) / (_vp.norm() * _vr.norm()))
                          if _vp.norm() > 0 and _vr.norm() > 0 else float("nan"))
                    rt._probe["rows"].append((pdb, arm_seed, i, float(_vp.norm()),
                                           float(_vr.norm()), _c, float(_cabs)))
                rt._probe["poses"] += 1
            _obj = cum / rt._step_poses_g[0]
            # THE RANKING TERM, as c_i * s_i. Its gradient is the exact gradient of the unit's
            # pairwise loss contribution, because dL/dtheta = sum_i (dL/ds_i)(ds_i/dtheta) and
            # c_i IS dL/ds_i. Divided by units-per-step so the term is a mean over units, the
            # way cum/accum_eff is a mean over poses.
            if _cs_src:
                import rank_loss as RKL
                for _p, _cv in _cs_src:
                    if not _cv:
                        continue
                    _sr = RKL.iptm_pooled(pred["pae"], feat_head, rt.configs,
                                          _asym_for(_p), pool=rt.a.rank_pool)
                    if _sr is None:
                        continue
                    _obj = _obj + (rt.a.alpha_rank * _cv / _cs_div) * _sr
                    # RECONSTRUCTED FOR THE LINEAR FORM ONLY. L = sum_i c_i s_i holds exactly
                    # there, so accumulating c_i * s_i gives the loss with no second
                    # evaluation. The SOFTPLUS loss is NOT sum_i c_i s_i -- its value comes
                    # from group_coeffs in pass A and is reported out of _xs_stat instead.
                    if not rt.a.rank_cross_seed:
                        rt._rank_seen["loss"] += float(_cv) * float(_sr.detach())
            _obj.backward()
            if _be1 is not None:
                _be1.record()
        # Explicit, though the return would free them anyway: this is the function's whole purpose,
        # and naming them stops a later edit from quietly hoisting one into the enclosing scope.
        del cum, met, pred, ld, x, plddt, pae, pde, resolved
        return out

    # THE RANKING COEFFICIENTS, FROM THE LABELS ALONE. NO EXTRA FORWARD PASS.
    #
    # The loss is LINEAR in the scores, L = sum_i c_i * s_i, so dL/ds_i does not contain any
    # score at all -- only DockQ:
    #
    #     c_i = dL/ds_i = - sum_{j valid} (d_i - d_j) / sum_{(i,j) valid} |d_i - d_j|
    #
    # Verified: the coefficients are byte-identical for score vectors as different as
    # [0.70,0.69,0.69,0.69,0.66], [0.1,0.9,0.5,0.2,0.7] and all-zeros.
    #
    # THIS REPLACES A TWO-PASS SCHEME, and the correction is worth recording. The first version
    # of this loss used softplus, whose derivative sigmoid(-m) DOES depend on every score, so a
    # no-grad forward was needed to collect them before the real one -- a third more GPU time
    # per pose, plus save/restore of the dropout RNG so the two passes evaluated the same
    # function, plus a per-pose scalar equality check to catch it when they did not. Switching
    # the loss to linear made every bit of that dead, and it was not removed at the time:
    # the surrounding machinery was not re-derived when the function it served changed.
    # Review of cd8e6ab7 caught it.
    #
    # Each pose's existing forward already yields its own s_i with a graph, so the ranking term
    # rides along on the backward that was happening anyway. One graph at a time, exact
    # gradient, zero extra forwards.
    _rank_c, _rank_on = {}, False
    if rt.a.rank_loss and not rt.a.loss_only and not rt.a.rank_cross_seed:
        import rank_loss as RKL
        _rk_pairs = sorted(rt._scored_asym.get(pdb) or ())
        _rk_lab = [(_p, rt._rank_lab.get(_rank_key(pdb, arm_seed, _p))) for _p in _rk_pairs]
        _rk_lab = [(_p, _d) for _p, _d in _rk_lab if _d]
        # AVERAGED OVER THE UNIT'S INTERFACES, so a 3-interface structure does not get three
        # times the weight of a 1-interface one.
        _nif = 0
        for _p, _dqmap in _rk_lab:
            _idx = [_i for _i in range(n_pose) if _i in _dqmap]
            if len(_idx) < 2:
                continue
            _cs, _np_ = RKL.coeffs([_dqmap[_i] for _i in _idx],
                                   weight_by_gap=True)
            if _np_ == 0:
                continue
            _nif += 1
            for _i, _cv in zip(_idx, _cs):
                _rank_c[(_i, _p)] = _rank_c.get((_i, _p), 0.0) + _cv
        if _nif:
            for _k in list(_rank_c):
                _rank_c[_k] /= float(_nif)
            _rank_on = True
            rt._rank_seen["units"] += 1

    for i in range(n_pose):
        _t_pose = time.perf_counter()
        # NO GRAPH UNDER --loss-only. Skipping only the backward would still build the
        # autograd graph for every pose and free it unused -- correct, but it pays the
        # activation memory of training for a measurement that needs none.
        # RESET BEFORE EVERY MEASURED POSE. max_memory_allocated is a HIGH-WATER MARK and the
        # only reset was once per epoch, so every pose after the first large one reported that
        # large one's number: the peak-versus-tokens curve was monotone in WALK ORDER, not a
        # function of the pose at all. The epoch's own peak is kept by hand below so the epoch
        # line still reports the epoch. Only under --mem-log, so the live run is untouched.
        #
        # OUTSIDE THE PASS BRANCH, NOT INSIDE THE else. It used to sit in the else, so poses
        # taken under no_grad -- every pass A pose, and all of them under --loss-only -- were
        # never reset while the MEM line below printed for them anyway. A whole step of pass A
        # lines then carried one stale number, identical on all of them, and a reader taking
        # max(peak) over the log was mixing a no-grad forward with a pose that backwards.
        _pass = "A" if (rt.a.loss_only or score_only) else "B"
        if rt.a.mem_log and torch.cuda.is_available():
            rt._mem_ep[0] = max(rt._mem_ep[0], torch.cuda.max_memory_allocated(rt.dev))
            torch.cuda.reset_peak_memory_stats(rt.dev)
        if rt.a.loss_only or score_only:
            with torch.no_grad():
                m = _one_pose(i)
        else:
            m = _one_pose(i)
        # PASS A'S POSES GET THEIR OWN BUCKET. Ticking them into pose_total would double-count
        # every pose and make the epoch accounting -- which is checked against the wall clock
        # and reported as a residual -- silently wrong.
        _tick("pose_a_total" if score_only else "pose_total", _t_pose)
        if rt.a.mem_log:
            if torch.cuda.is_available():
                rt._mem_ep[0] = max(rt._mem_ep[0], torch.cuda.max_memory_allocated(rt.dev))
            # AT A CONSISTENT BOUNDARY: after backward, and after the pose's tensors are gone.
            # allocated = live tensors. reserved = what the allocator holds, which is what
            # nvidia-smi shows -- which is why a high nvidia-smi number proves nothing by itself.
            # THE PASS IS PART OF THE MEASUREMENT. A pass A pose runs under no_grad and
            # stores no activations, so its peak is not comparable to a pass B pose's and is
            # not what any checkpointing threshold is decided from.
            print(f"MEM {pdb}/{arm_seed} pose {i} pass {_pass} tokens {n_tok}: "
                  f"allocated {torch.cuda.memory_allocated(rt.dev) / 2**30:.2f} GB  "
                  f"peak {torch.cuda.max_memory_allocated(rt.dev) / 2**30:.2f}  "
                  f"reserved {torch.cuda.memory_reserved(rt.dev) / 2**30:.2f}", flush=True)
        yield m
