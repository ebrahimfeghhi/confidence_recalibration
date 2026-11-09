"""`trainer/dataset.py` -- which units, interfaces and labels THIS run trains on.

EXTRACTED from `phases/phase_k_train.py` lines 1936-2905 (source `d34baebc`).

Per the review on issue #9 this is run POLICY, not a reusable data reader: given this
run's config it answers which units, seeds, interfaces and labels are eligible, in what
order, and what evidence establishes they match. Reusable primitives (read_manifest,
resolve_chain_mapping, load_labels, compute_fingerprint) belong in `data/` and are not
moved here.

LEAN: 14 lines of branches for flags this copy does not carry (--l2-sp, LoRA,
crop, early stop, the probes, --drop-pde) are dropped rather than ported, at the owner's
direction. Every refusal that guards a flag this copy DOES carry is kept verbatim --
several exist because the failure already happened here, so softening one would delete a
lesson rather than a check.
"""
import math
import numpy as np
import sys
# THE HELPERS main() DECLARED INLINE. Imported by name rather than with a star
# so the dependency is visible and a missing one is an ImportError here.
from helpers import (_check_eval_pool, _pae_mask_check, _rank_key)
import csv
import json
import os

import torch


def resolve_training_dataset(rt, a, is_main=True):
    """Returns the resolved manifest this run trains on."""
    loss_fn = rt.ProtenixLoss(rt.configs)
    _dropped_terms = set()
    if a.pae_only:
        # DO NOT COMPUTE THE ALL-ATOM LDDT WE THEN THROW AWAY.
        #
        # ProtenixLoss computes per_atom_lddt unconditionally -- `if all(x in pred_dict for x in
        # ["plddt","pde","pae","resolved"])`, which is always true because the head returns all
        # four -- and those values feed ONLY the plddt_loss lambda, which --pae-only stubs to
        # zeros. So it is pure waste. MEASURED on an idle card: 3.4 ms per pose at 11,504 atoms,
        # about 1% of the 0.300 s/pose, but up to 1.66 GB of TRANSIENT allocation, which is the
        # actual reason to remove it -- that headroom is what a larger crop or a larger accum
        # would need. Found in review.
        #
        # Patched at the module where loss.py looks it up, and ONLY under --pae-only, where the
        # consumer is already a stub that ignores both return values. The zeros are shaped from
        # pred_coordinate so nothing downstream can trip over them if it ever does read.
        import protenix.model.loss as _PL
        _orig_lddt = _PL.calculate_atom_bespoke_lddt

        def _no_lddt(pred_coordinate, true_coordinate, *_args, **_kw):
            z = torch.zeros(pred_coordinate.shape[:-1], dtype=pred_coordinate.dtype,
                            device=pred_coordinate.device)
            return z, z
        _PL.calculate_atom_bespoke_lddt = _no_lddt
        print("  pae-only: all-atom lddt SKIPPED (feeds only the stubbed pLDDT term); "
              "measured 3.4 ms and up to 1.66 GB per pose")
        # SAME STUB TRICK as --drop-pde, for the same reason: a zero weight still runs the module
        # and still pays for it. Replacing the module is what actually removes the term.
        #
        # IT ANSWERS ANY ENTRY POINT, not just forward(). ProtenixLoss does NOT call these three
        # uniformly: pde_loss and resolved_loss go through __call__, but pLDDT is reached by a
        # NAMED METHOD -- `self.plddt_loss.forward_given_atom_lddt(...)` (loss.py:1771). A stub
        # carrying only forward() raised AttributeError at pose 0 of unit 0, so the first version
        # of this flag could not run a single step. Generalising from --drop-pde, which happens to
        # be the easy case, is what produced that.
        class _Zero(torch.nn.Module):
            def forward(self, *_args, **_kw):
                return torch.zeros((), device=rt.dev)

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    if name.startswith("_"):
                        raise
                    return self.forward

        # CHECKED AGAINST THE REAL MODULE, not with hasattr on the stub. hasattr(loss_fn, name)
        # only proved the attribute existed; it said nothing about the calling convention, which
        # is precisely what broke. This asserts the stub answers every public callable the module
        # being replaced exposes, so a library that grows a new named entry point fails loudly.
        for _name in ("plddt_loss", "pde_loss", "resolved_loss"):
            _orig = getattr(loss_fn, _name, None)
            if _orig is None:
                raise SystemExit(f"STOP: --pae-only cannot drop {_name}: ProtenixLoss has no such "
                                 f"submodule. It has {sorted(k for k in dir(loss_fn) if 'loss' in k)}. "
                                 f"Dropping nothing while announcing PAE-only would mislabel the run.")
            _stub = rt._Zero()
            _base = set(dir(torch.nn.Module))
            _api = [m for m in dir(type(_orig))
                    if not m.startswith("_") and m not in _base and callable(getattr(_orig, m, None))]
            _missing = [m for m in _api if not callable(getattr(_stub, m, None))]
            if _missing:
                raise SystemExit(f"STOP: the --pae-only stub cannot stand in for {_name}: it does "
                                 f"not answer {_missing}, which {type(_orig).__name__} exposes and "
                                 f"the loss may call. A stub that is missing the entry point the "
                                 f"caller uses dies mid-run, or worse, silently skips a term.")
            setattr(loss_fn, _name, _stub)
            _dropped_terms.add(_name)
            print(f"  dropped {_name} ({type(_orig).__name__}); stub answers {sorted(_api)}")
        print("PAE-ONLY: pLDDT, PDE and resolved DROPPED -- not computed, not trained. "
              "Every gradient now goes to the one output ipTM reads.")
    # "m" is the scored-cross-chain pair mask; "w" the per-anchor-row multiplier --pae-row-frac
    # installs; "bins"/"pm" are stashed by _masked_calc on diagnostic poses only, so the per-row
    # loss can be read back AFTER the forward without recomputing the label.
    _pae_keep = {"m": None, "wm": None, "blocks": None, "bins": None, "pm": None,
                 "stash": False, "pdb": None, "unit": None, "replay": None, "unw_label": None}
    _scored_asym = {}
    _pae_mask_stats = {"masked": 0, "unmasked": 0}
    if a.emit_iptm_rows and not a.loss_only:
        raise SystemExit(
            "STOP: --emit-iptm-rows is only supported with --loss-only. rank_loss.iptm is the "
            "DIFFERENTIABLE scorer -- 'minus its torch.no_grad' -- so in a grad-enabled run it "
            "builds an fp32 [1,N,N,64] softmax that stays reachable from the loop variable across "
            "calculate_losses and backward, at the memory peak (2.96 GiB at --crop-size 3400). "
            "--loss-only already wraps the pose in torch.no_grad, which is why it is safe there.")
    _iface_of = {}
    # LAZILY OPENED, so a run without --emit-iptm-rows touches no file. Rows are flushed per unit
    # rather than held to the end: a measurement pass that dies on its last unit should still
    # leave the rows it earned, the same reason the subprocess trajectories are the durable record.
    _iptm_fh = {"fh": None, "w": None, "n": 0}
    if a.pae_scored_pairs:
        import csv as _csv
        _roles_p = a.pae_scored_roles or os.path.join(a.fold, "chain_roles.csv")
        _roles = {}
        with open(_roles_p) as _fh:
            for _r in _csv.DictReader(_fh):
                _roles.setdefault(_r["pdb"], {})[_r["ref_chain"]] = _r["model_chain"]
        # Model chains are lettered A, B, C... in asym_id order by write_job_cif, so the letter's
        # position in the alphabet IS the asym index -- the same join phase_k_eval does.
        _miss = []
        with open(a.pae_scored_pairs) as _fh:
            for _r in _csv.DictReader(_fh):
                _m1 = _roles.get(_r["pdb"], {}).get(_r["chain1"])
                _m2 = _roles.get(_r["pdb"], {}).get(_r["chain2"])
                if _m1 is None or _m2 is None:
                    _miss.append(f"{_r['pdb']}:{_r['chain1']}/{_r['chain2']}")
                    continue
                # NORMALISED LOW-HIGH. chain_roles maps ref_chain -> model_chain, and the
                # pool's (chain1, chain2) order is the REFERENCE order, so the model-chain pair
                # comes out descending for most interfaces -- 46 of 77 on the overfit30 pool.
                # _set_pae_mask symmetrises explicitly so it never cared, but --rank-loss looks up
                # DockQ labels by this tuple and rank_loss.parse_iface normalises to (min, max):
                # unnormalised, the lookup found 9 of 77 pairs and the other 68 interfaces would
                # have paid a forward for zero ranking gradient. chain_pair_iptm is symmetric --
                # phase_k_eval reads cpi[ai, aj] in the pose's own order and reproduces the fold's
                # stored ipTM to 2e-3 on every row, descending ones included -- so normalising
                # changes no value, only the key.
                _a1 = ord(_m1) - ord("A")
                _a2 = ord(_m2) - ord("A")
                _scored_asym.setdefault(_r["pdb"], set()).add((min(_a1, _a2), max(_a1, _a2)))
                # THE POOL'S OWN REFERENCE NAMING, kept so an emitted row can carry the `iface`
                # string results.csv uses. The asym pair is a MODEL-chain index pair and says
                # nothing about which reference chains it came from.
                _iface_of[(_r["pdb"], (min(_a1, _a2), max(_a1, _a2)))] = \
                    f"{_r['pdb']}_{_r['chain1']}{_r['chain2']}"
        if _miss:
            raise SystemExit(f"STOP: {len(_miss)} pool interface(s) have no role mapping, e.g. "
                             f"{_miss[:4]}. The PAE mask would silently drop their blocks.")
        _orig_calc = loss_fn.pae_loss.calculate_label

        def _masked_calc(*args, **kw):
            # REPLAY SERVES THE SECOND, UNWEIGHTED FORWARD of the same pose without recomputing
            # the label. calculate_label does the frame alignment; only the cross-entropy needs
            # redoing to read the unweighted number, and that number has to be per-pose rather
            # than sampled so trajectory.csv stays comparable with every unweighted arm.
            rp = _pae_keep["replay"]
            if rp is not None:
                return rp
            sq, bins, pm = _orig_calc(*args, **kw)
            m = _pae_keep["m"]
            if m is not None:
                pm = pm * m.to(pm.dtype)
            # THE WEIGHTED MASK MUST BE FLOAT, EXPLICITLY. PAELoss reduces as
            # sum(loss * pair_mask) / sum(pair_mask) -- already a weighted mean -- so a real-valued
            # mask is a weighted PAE term with no change to Protenix. But calculate_label returns a
            # BOOL pair_mask, so the earlier `w.to(pm.dtype)` cast every weight to True and the
            # whole term was a silent no-op that still logged itself as weighted. Caught in review;
            # test_pae_row_weight.py now asserts the weighted mask DIFFERS from the unweighted one.
            wm = _pae_keep["wm"]
            if wm is not None:
                pm = pm.to(torch.float32) * wm.to(torch.float32)
            _pae_keep["unw_label"] = (sq, bins, (pm > 0).to(torch.float32) if wm is not None
                                      else pm)
            if _pae_keep["stash"]:
                _pae_keep["bins"], _pae_keep["pm"] = bins, pm
            return sq, bins, pm

        loss_fn.pae_loss.calculate_label = _masked_calc
        print(f"PAE term MASKED to scored chain pairs: {len(_scored_asym)} structure(s), "
              f"{sum(len(v) for v in _scored_asym.values())} interface(s), from "
              f"{a.pae_scored_pairs}")

    # ------------------------------------------------------------------ ROW-WEIGHTED PAE TERM
    # WHY THIS EXISTS. ipTM is max_i t_i over the anchor rows, where t_i is row i's mean predicted
    # cross-chain TM weight. Selection therefore reads ONE row of ~332 while the PAE term grades
    # every row equally. This reweights the term toward the rows selection can read.
    #
    # WHAT IT DOES NOT DO, and this is the reason to prefer it to a ranking loss: it never changes
    # the LABEL. Every pair is still trained toward its own true error bin, so the term cannot be
    # lowered by distorting ipTM away from calibration, and it stays sensitive to the absolute
    # confidence level that a softmax ranking loss is blind to by construction.
    #
    # THE WEIGHTS ARE DETACHED. If they carried gradient the model could lower the loss by moving
    # weight onto rows it is already good at instead of getting better.
    if float(0.0) > 0:
        if not a.pae_scored_pairs:
            raise SystemExit("STOP: --pae-row-frac needs --pae-scored-pairs. t_i is a mean over a "
                             "row's SCORED cross-chain targets; with no pool there is no cross "
                             "block to average over and the ranking of rows would be meaningless.")
        if not 0.0 < float(0.8) < 1.0:
            raise SystemExit(
                f"STOP: --pae-row-mass must be in (0, 1), got {0.8}. At 1.0 every row "
                f"outside the top set gets weight 0, which is a hard top-k mask: those rows are "
                f"then unsupervised and free to drift over-confident, and since ipTM is a max an "
                f"over-confident row is exactly the one selection reads. Pass 0.8 for the "
                f"16x-focus-with-a-floor this flag is for.")
        if not 0.0 < float(0.0) <= 1.0:
            raise SystemExit(f"STOP: --pae-row-frac must be in (0, 1], got {0.0}.")
        if int(10) < 1:
            raise SystemExit(f"STOP: --pae-row-floor must be >= 1, got {10}.")
        # THE MATH LIVES IN phases/pae_row_weight.py so a test can import the same code this
        # run uses. run_logging/test_pae_row_weight.py asserts max(row_t(...)) equals Protenix's
        # own calculate_iptm, and that frac 1.0 reproduces the unweighted term bit for bit.
        sys.path.insert(0, os.path.dirname(os.path.abspath(rt.__file__)))
        import pae_row_weight as _rw
        _orig_pae_fwd = loss_fn.pae_loss.forward
        _pl = loss_fn.pae_loss
        _rf, _rm, _rfl = float(0.0), float(0.8), int(10)
        _row_seen = {"n": 0}
        _row_diag_p = os.path.join(a.out, f"pae_rows_rank{rank}.csv")
        _row_diag_fh = {"fh": None, "w": None}
        _pae_unw = {"v": float("nan")}

        def _unweighted(kw):
            """The m-masked but row-UNWEIGHTED loss for this pose, every pose.

            WHY EVERY POSE. trajectory.csv's pae_loss column is whatever the trained term returns,
            and that is now a reweighted mean. A mean-1 weight does NOT preserve the value unless
            the per-row loss is uncorrelated with the weight -- and the weight is chosen BY t_i,
            which the rho column exists to show is correlated. So the weighted number is not
            comparable with any unweighted arm, and a 1-in-50 sample is not a curve. The label is
            replayed rather than recomputed, so this costs one cross-entropy, not an alignment.
            """
            lab = _pae_keep["unw_label"]
            if lab is None:
                return float("nan")
            try:
                with torch.no_grad():
                    _pae_keep["replay"] = lab
                    return float(_orig_pae_fwd(**kw))
            except Exception as _e:
                print(f"  WARNING: unweighted PAE readback failed "
                      f"({type(_e).__name__}: {_e})", flush=True)
                return float("nan")
            finally:
                _pae_keep["replay"] = None

        def _row_log(pdb, unit, n, weighted, unw, kw):
            """Realised effective sample size, t_i quantiles, and the rank correlation between
            t_i and that row's own PAE loss.

            THE CORRELATION IS THE MEASUREMENT. If the high-t_i rows are already the best-fit ones
            then this weighting has no headroom, and that is answerable from the first epoch rather
            than from DockQ sixty epochs later.
            """
            try:
                with torch.no_grad():
                    bins, pm = _pae_keep["bins"], _pae_keep["pm"]
                    rho, rl_mean, ess, k_real = [float("nan")] * 3 + [0]
                    tq = [float("nan")] * 3
                    if bins is not None and pm is not None:
                        ess = _rw.effective_sample_size(pm)
                        hf = kw["has_frame"].reshape(-1).bool()
                        # CHUNKED for the same reason row_t is: a full fp32 log_softmax over the
                        # crop is ~555 MB at our largest, inside the live autograd graph.
                        lg = kw["logits"]
                        while lg.dim() > 4:
                            lg = lg[0]
                        lg = lg[:, hf] if lg.dim() == 4 else lg[hf].unsqueeze(0)
                        bq = bins if bins.dim() == lg.dim() else bins.unsqueeze(0)
                        nrow = int(hf.sum())
                        ce = torch.zeros(nrow, lg.shape[-2], device=lg.device,
                                         dtype=torch.float32)
                        for lo in range(0, nrow, _rw.ROW_CHUNK):
                            hi = min(lo + _rw.ROW_CHUNK, nrow)
                            c = lg[:, lo:hi].float()
                            ce[lo:hi] = -(bq[:, lo:hi].float()
                                          * torch.log_softmax(c, dim=-1)).sum(-1).mean(0)
                            del c
                        base = (pm > 0).to(ce.dtype)     # the unweighted support of the same mask
                        cnt = base.sum(-1)
                        rl = (ce * base).sum(-1) / cnt.clamp_min(1.0)
                        ok = cnt > 0
                        k_real = int(ok.sum())
                        if k_real > 2:
                            tt = pm.to(torch.float64).sum(-1)      # realised row weight
                            def _rk(x):
                                return torch.argsort(torch.argsort(x)).to(torch.float64)
                            x, y = _rk(tt[ok]), _rk(rl[ok].double())
                            x, y = x - x.mean(), y - y.mean()
                            den = float(x.norm() * y.norm())
                            if den > 0:
                                rho = float((x * y).sum() / den)
                            q = torch.tensor([0.5, 0.9, 1.0], dtype=torch.float64,
                                             device=tt.device)
                            tq = [float(v) for v in torch.quantile(tt[ok], q)]
                        rl_mean = float(rl[ok].mean()) if k_real else float("nan")
                row = {"pdb": pdb, "unit": unit, "n_rows": n, "n_scored_rows": k_real,
                       "ess": round(ess, 2), "pae_weighted": round(float(weighted), 6),
                       "pae_unweighted": round(float(unw), 6),
                       "row_loss_mean": round(rl_mean, 6),
                       "rho_rowweight_vs_rowloss": round(rho, 6),
                       "rw_p50": round(tq[0], 6), "rw_p90": round(tq[1], 6),
                       "rw_max": round(tq[2], 6)}
                if _row_diag_fh["fh"] is None:
                    _row_diag_fh["fh"] = open(_row_diag_p, "a", newline="")
                    _row_diag_fh["w"] = csv.DictWriter(_row_diag_fh["fh"], fieldnames=list(row))
                    if _row_diag_fh["fh"].tell() == 0:
                        _row_diag_fh["w"].writeheader()
                _row_diag_fh["w"].writerow(row)
                _row_diag_fh["fh"].flush()
            except Exception as _e:
                # NAMED, NOT SWALLOWED. An OOM here used to leave pae_rows_rank*.csv silently
                # empty while the launch note called it the diagnostic that answers the premise.
                print(f"  WARNING: row diagnostic failed for {pdb}/{unit} "
                      f"({type(_e).__name__}: {_e})", flush=True)

        def _row_weighted_fwd(**kw):
            hf = kw["has_frame"].reshape(-1).bool()
            n = int(hf.sum())
            _row_seen["n"] += 1
            diag = (int(50) > 0
                    and _row_seen["n"] % int(50) == 0)
            _pae_keep["unw_label"] = None
            blocks = _pae_keep["blocks"]
            if n == 0 or not blocks:
                _pae_keep["wm"] = None
                return _orig_pae_fwd(**kw)
            with torch.no_grad():
                _pae_keep["wm"] = _rw.weighted_mask(
                    kw["logits"], hf, blocks, _rf, _rm, _rfl, _pl.min_bin, _pl.max_bin)
            _pae_keep["stash"] = bool(diag)
            try:
                out = _orig_pae_fwd(**kw)
            finally:
                _pae_keep["wm"], _pae_keep["stash"] = None, False
            unw = _unweighted(kw)
            _pae_unw["v"] = unw
            if diag:
                _row_log(_pae_keep.get("pdb"), _pae_keep.get("unit"), n, out, unw, kw)
            _pae_keep["bins"] = _pae_keep["pm"] = _pae_keep["unw_label"] = None
            return out

        loss_fn.pae_loss.forward = _row_weighted_fwd
        _bk = min(332, max(_rfl, -(-int(_rf * 332) // 1)))
        _bw = _rw.row_weights(torch.arange(332, dtype=torch.float32), None, _rf, _rm, _rfl)[0]

    # RANKING LOSS SETUP. Refused rather than silently degraded when its inputs are absent: without
    # --pae-scored-pairs there is no interface to read ipTM for, and without labels there is no
    # ordering to learn. Both failures would otherwise train the PAE term alone and look like a
    # ranking experiment.
    # UNITS PER STEP, so the ranking term is a mean over units the way cum/accum_eff is a mean
    # over poses. accum_eff counts poses; each unit contributes --nsample of them.
    _rank_lab = {}
    _rank_div = max(1, rt.accum_eff // max(1, a.nsample))
    _rank_seen = {"loss": 0.0, "units": 0}
    # CROSS-SEED STATE. _xs_coef holds dL/ds for every pose of the CURRENT step, keyed
    # (pdb, seed, sample, pair) -- a pose's coefficient depends on every other pose of its
    # interface, which is why it cannot be built per unit the way the linear one is. _xs_div is the
    # number of interface-groups in the step, so the term is a mean over groups; it is a one-element
    # list because pass A writes it and the pose loop reads it from a different frame.
    # RELATIVE-iTM LABEL CACHE, loaded lazily per structure and kept. One <pdb>.npz is a few KB
    # and a structure recurs every epoch, so reading it once per process is right; loading all 368
    # at startup would be wasted on a rank that never sees most of them.
    _rel_cache = {}
    _rel_stat = {"groups": 0, "poses": 0, "short_groups": 0, "csum": 0.0, "missing_files": 0}

    # THE SAME MODULE phase_k_eval USES, so the per-pose chain join is literally the evaluator's
    # and cannot drift from it.
    import phase_k_eval as _PKE
    # THE SAME ROLES FILE THE PAE MASK USES. Defined here rather than reaching for _roles_p
    # because that name is bound inside `if a.pae_scored_pairs:` and this must be safe either way.
    _rel_roles_p = a.pae_scored_roles or os.path.join(a.fold, "chain_roles.csv")
    _rel_refs = (_PKE.pool_ref_pairs(a.pae_scored_pairs)
                 if (a.rank_form == "rel" and a.pae_scored_pairs) else {})
    # _roles_p, NOT <fold>/chain_roles.csv. Raised in review and correct: --pae-scored-roles
    # already steered the PAE mask and RKL.load_labels at line 1769, but these two calls still
    # hardcoded the default file. chain_roles.csv names 501 structures and interface_chain_pairs
    # SKIPS an interface whose roles it cannot find, so under --rank-form rel every wave-1
    # structure would have had no _rel_ifaces while pass A built _pose_pairs from the full pool --
    # surfacing as _exp != _got, or as silently incomplete rel diagnostics. chain_roles_all.csv
    # covers 853 of 853; the point of the flag is that both readers use it.
    _rel_ifaces = {_p: [t[0] for t in _l] for _p, _l in
                   ((_PKE.interface_chain_pairs(a.pae_scored_pairs, _rel_roles_p)[0]).items()
                    if (a.rank_form == "rel" and a.pae_scored_pairs) else {}.items())}
    # IFACE -> STATIC ASYM PAIR, for the DockQ lookup ONLY. _rank_lab is keyed by the asym pair
    # rank_loss.parse_iface derives from chain_roles, while rel keys everything by the interface
    # name. DockQ is a property of the INTERFACE -- results.csv is keyed by its `iface` column --
    # so which asym representation indexes it is an internal detail and the static one is
    # correct here. This is NOT the same question as which pair to SCORE, which is per pose.
    _rel_dqkey = {}
    if a.rank_form == "rel" and a.pae_scored_pairs:
        for _p2, _l2 in _PKE.interface_chain_pairs(
                a.pae_scored_pairs, _rel_roles_p)[0].items():
            for _ifc2, _x2, _y2 in _l2:
                _rel_dqkey[_ifc2] = (min(_x2, _y2), max(_x2, _y2))

    def _spearman(xs, ys):
        """Rank correlation without scipy: rank both, then Pearson. Ties get average ranks."""
        n = len(xs)
        if n < 3:
            return None

        def rank(v):
            order = sorted(range(n), key=lambda k: v[k])
            r = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                    j += 1
                avg = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    r[order[k]] = avg
                i = j + 1
            return r

        rx, ry = rank(xs), rank(ys)
        mx, my = sum(rx) / n, sum(ry) / n
        sx = sum((v - mx) ** 2 for v in rx) ** 0.5
        sy = sum((v - my) ** 2 for v in ry) ** 0.5
        if sx == 0 or sy == 0:
            return None
        return sum((a_ - mx) * (b_ - my) for a_, b_ in zip(rx, ry)) / (sx * sy)

    _ost_cache = {}

    def _pose_pairs(pdb, lab_p, samp):
        """[(iface, ax, ay)] for ONE pose, from OST -- the SAME join phase_k_eval scores with.

        WHY NOT THE STATIC PAIRS. --rank-form rel's labels are computed on the pose's OST-resolved
        pair; scoring the static pair and looking the label up by interface name would pair
        predicted ipTM for one chain pair with true iTM for another. Measured: the two mappings
        disagree on 1,095 of 18,400 (structure, seed, pose) checks across 52 of 368 structures,
        and 7stf resolves 7stf_BC to (3,1) statically against (3,0) per pose -- a different chain,
        not a reordering. The first version of this arm had exactly that mismatch and no coverage
        check could see it, because the keys still matched. Caught in review.

        REFUSES rather than falling back. A fallback to the static pair is the defect itself.
        """
        import phase_j_build_cache as _BC
        key = (pdb, os.path.basename(lab_p), int(samp))
        if key in _ost_cache:
            return _ost_cache[key]
        sd = _rank_key(pdb, os.path.basename(lab_p)[:-4], (0, 0))[1]
        try:
            ud = _BC.seed_dir(a.fold, pdb, int(sd))
        except Exception as _e:                              # noqa: BLE001
            raise SystemExit(
                f"STOP: --rank-form rel cannot resolve a unit directory for {pdb}/{sd} "
                f"({type(_e).__name__}), so this pose's chains cannot be mapped the way its "
                f"label was built.")
        got, why = _PKE.ost_chain_pairs(ud, pdb, int(samp), _rel_refs.get(pdb) or [])
        if got is None:
            raise SystemExit(
                f"STOP: --rank-form rel has no OST chain mapping for {pdb}/{sd} sample {samp} "
                f"({why}). Falling back to the static pair is the exact mismatch this resolution "
                f"exists to prevent.")
        _ost_cache[key] = sorted(got)
        return _ost_cache[key]

    def _rel_labels(pdb):
        """{(seed, sample, ax, ay): g} for one structure, from --rel-labels.

        A MISSING FILE IS FATAL, not empty. An empty dict would make every group "short", the loop
        would skip them all, and the run would train on PAE alone while its log and saved guard
        both said --rank-form rel -- the row-weight bool-cast failure in another costume.
        """
        if pdb in _rel_cache:
            return _rel_cache[pdb]
        if not a.rel_labels:
            raise SystemExit("STOP: --rank-form rel needs --rel-labels; build it with "
                             "run_logging/build_itm_labels.py")
        f = os.path.join(a.rel_labels, f"{pdb}.npz")
        if not os.path.exists(f):
            _rel_stat["missing_files"] += 1
            raise SystemExit(
                f"STOP: no relative-iTM label file for {pdb} at {f}. Every training structure "
                f"needs one; rebuild with run_logging/build_itm_labels.py and check its manifest "
                f"covers this manifest-selection.")
        z = np.load(f, allow_pickle=True)
        col = "g_max" if a.rank_pool == "max" else "g_mean"
        # THE LABEL MUST MATCH THE POOLING. Training the mean-pooled score toward a max-pooled
        # label would regress onto a different quantity than the one being produced, and both
        # columns exist precisely so this is a lookup rather than a conversion.
        if "iface" not in z.files:
            raise SystemExit(
                f"STOP: {f} is an OLD label cache keyed by an asym pair. Those caches paired a "
                f"value computed on the pose's OST pair with the STATIC pair as its key, which is "
                f"the mismatch review caught. Rebuild with run_logging/build_itm_labels.py.")
        d = {(int(sd), int(sp), str(ifc)): float(g)
             for ifc, sd, sp, g in zip(z["iface"], z["seed"], z["sample"], z[col])}
        _rel_cache[pdb] = d
        return d

    _xs_coef, _xs_div, _xs_stat = {}, [1], {"groups": 0, "pairs": 0, "loss": 0.0, "steps": 0,
                                             "last_groups": 0, "skipped": 0}
    # THE CURRENT STEP'S EXPECTED POSE COUNT. accum_local is a constant only when every
    # step holds the same number of units, which --rank-cross-seed breaks: S structures x
    # their seeds need not divide `world`, so the short rank rotates and the quota varies
    # by step. Driving the boundary from the quota keeps every rank stepping on the same
    # iteration -- a mismatch there HANGS on the first all-reduce rather than failing.
    _step_poses = [rt.accum_local]
    # AND THE GLOBAL COUNT, which is what the PAE term's mean is taken over. accum_eff is
    # K*nsample*world = 90 here, but a cross-seed step holds S*seeds*nsample = 100 poses across the
    # ranks, so dividing by accum_eff would scale the PAE gradient by 100/90. Set once per epoch
    # from the actual step shape; equal to accum_eff when --rank-cross-seed is off.
    _step_poses_g = [rt.accum_eff]
    # THE COSINE HORIZON, in optimiser steps, filled in just before train_loop from the same
    # `_expect` the startup banner prints. A list because train_loop closes over it and the value
    # is not known until the unit list has been flattened and sharded.
    sys.path.insert(0, os.path.dirname(os.path.abspath(rt.__file__)))
    import lr_schedule as _LRS
    _lr_horizon = [0]
    # STEPS PER EPOCH THE HORIZON WAS CALIBRATED FROM, 0 when cross-seed packing is not in use.
    # build_epoch_cross_seed compares its own packing against it every epoch.
    _xs_spe = [0]
    _rank_tau = float(None) if None is not None else None
    _probe = {"rows": [], "units": 0, "poses": 0, "sum": {}, "finite": {}, "nonzero": {}}
    if a.rank_loss and a.loss_only:
        # SAME REASON. Nothing below is needed for a measurement, and --rank-labels points at the
        # training fold while this subprocess reads the validation one.
        print("  --loss-only: ranking setup skipped (the term does not run in a measurement)")
    elif a.rank_loss:
        import rank_loss as RKL
        if not _scored_asym:
            raise SystemExit("STOP: --rank-loss needs --pae-scored-pairs. The ranking term reads "
                             "chain_pair_iptm for a NAMED interface; with no scored pair there is "
                             "nothing to rank and no join to the DockQ labels.")
        _rl_p = None or os.path.join(a.fold, "results.csv")
        if not os.path.exists(_rl_p):
            raise SystemExit(f"STOP: --rank-loss found no DockQ labels at {_rl_p}. Pass "
                             f"--rank-labels explicitly.")
        _rank_lab, _rl_bad = RKL.load_labels(_rl_p, _roles_p)
        if not _rank_lab:
            raise SystemExit(f"STOP: {_rl_p} yielded no usable (pdb, seed, pair) -> DockQ groups.")
        # THE LABELS MUST COVER THE INTERFACES THIS RUN TRAINS, checked here and not at the first
        # pose. A structure whose labels are missing would contribute zero ranking gradient while
        # still paying a full forward, so the arm would be a partial experiment reported as a whole
        # one -- the same class of error as the LoRA bundles that trained four epochs with no
        # held-out points.
        # COMPLETENESS, NOT MERE PRESENCE. The first version checked only that a scored
        # structure appeared SOMEWHERE in the labels, which a structure with one labelled seed out
        # of ten would pass -- and its unlabelled units would then pay a forward and contribute no
        # ranking gradient, making this a partial experiment reported as a whole one. That is the
        # same class as the LoRA bundles that trained four epochs with zero held-out points.
        # Review of cd8e6ab7 pointed out the guard was weaker than the doc claimed. So every
        # (pdb, seed, interface) this run will train must carry --nsample finite DockQ values at
        # the expected sample indices.
        _rl_pdbs = {k[0] for k in _rank_lab}
        _rl_missing = sorted(set(_scored_asym) - _rl_pdbs)
        import rank_loss as _RKL_cfg
        _tau_eff = _rank_tau if _rank_tau is not None else _RKL_cfg.DEFAULT_TAU
        if a.rank_form == "rel":
            # The refusals for this form run EARLY, before any hardware -- see the block beside
            # the LAYERNORM_TYPE check. This is only the report.
            _man = os.path.join(a.rel_labels, "manifest.json")
            _mj = json.load(open(_man)) if os.path.exists(_man) else {}
            print(f"RELATIVE iTM REGRESSION on chain_pair ipTM pooled by {a.rank_pool.upper()}: "
                  f"L = (1/P) sum_i [(s_i - mean_s) - (g_i - mean_g)]^2 per interface over all "
                  f"its seeds, alpha {a.alpha_rank:g}"
                  + (", STANDARDISED by the within-interface std" if a.rel_standardize else "")
                  + f". Labels {a.rel_labels} ({_mj.get('structures', '?')} structure(s), "
                  f"{_mj.get('rows', '?')} row(s), column "
                  f"{'g_max' if a.rank_pool == 'max' else 'g_mean'}, digest "
                  f"{str(_mj.get('digest', '?'))[:12]}). The coefficients sum to EXACTLY zero, so "
                  f"this term cannot move an interface's level and its accumulated gradient "
                  f"cancels by construction -- alpha must come from --grad-probe norms, not from "
                  f"loss values.")
        print(f"RANKING LOSS on chain_pair_iptm: form {a.rank_form}, alpha {a.alpha_rank:g}, "
              f"weight_by_gap={True}, {len(_rank_lab)} labelled "
              f"(pdb, seed, interface) group(s) from {_rl_p}"
              + (f", {_rl_bad} unparsed row(s)" if _rl_bad else ""))
        # VALIDATED BEFORE ANYTHING IS PRINTED. tau <= 0 is refused in three places while these
        # two floats took nan, inf and negatives silently: --rank-topk-decay nan makes every
        # weight nan and the `> 0` banner test below is FALSE for nan, so the log would not even
        # mention the decay while the loss trained on nan; --rank-within-weight -1 drives den
        # negative and flips the loss's sign.
        for _nm, _vv in (("--rank-within-weight", 1.0),
                         ("--rank-topk-decay", 0.0),
                         ("--rank-straddle", 0.0)):
            if not math.isfinite(_vv) or _vv < 0:
                raise SystemExit(f"STOP: {_nm} {_vv!r} must be a finite number >= 0.")
        # THE BANNERS MUST NOT DESCRIBE A PATH THAT IS NOT TAKEN. These three flags are read only
        # by group_coeffs on the pass-A cross-seed path. The linear form without --rank-cross-seed
        # goes through RKL.coeffs, which takes labels alone and honours none of them, and the
        # expected form returns before the pair loop. Printing "CROSS-SEED pairs only -- the
        # deployed objective" while training within-seed linear pairs is how a run gets written up
        # as the wrong experiment, so a non-default setting on a path that ignores it is refused
        # rather than announced.
        # THE TWO GROUP FORMS IGNORE THEM TOO, for the same reason the expected form does: they
        # are defined on a whole ordering, so there is no pair to include, weight or decay. Added
        # to this guard rather than documented, because the banner below would otherwise print a
        # pair family for a loss that has none.
        _group_form = a.rank_form in ("spearman", "distance", "spearman_distance")
        _ignores = (a.rank_form == "expected") or _group_form or not a.rank_cross_seed
        _nondefault = [f"--rank-pairs {'cross'}" for _ in (1,) if 'cross' != "cross"] \
            + [f"--rank-within-weight {1.0:g}" for _ in (1,)
               if 1.0 != 1.0] \
            + [f"--rank-topk-decay {0.0:g}" for _ in (1,) if 0.0 != 0.0] \
            + [f"--rank-straddle {0.0:g}" for _ in (1,) if 0.0 != 0.0]
        if _ignores and _nondefault:
            raise SystemExit(
                "STOP: " + ", ".join(_nondefault) + " cannot take effect here. "
                + (f"--rank-form {a.rank_form} is defined on the whole ordering of a group, not "
                   "on pairs, so the family, the within-seed weight, the rank decay and the "
                   "straddle filter are all inapplicable. Its own knobs are "
                   "--rank-spearman-tau, --rank-dist-weight and --rank-rel-scale."
                   if _group_form else
                   "--rank-form expected is pair-free -- group_coeffs returns before the pair "
                   "loop, so the family, the within-seed weight and the rank decay are all "
                   "inapplicable."
                   if a.rank_form == "expected" else
                   "without --rank-cross-seed the loss goes through the per-unit path "
                   "(rank_loss.coeffs), which takes labels alone and honours none of these three. "
                   "Add --rank-cross-seed, or drop these flags.")
                + " Refused rather than warned: the startup banner would otherwise name an "
                  "objective the run is not training.")

        # THE FAMILY IS NAMED BEFORE THE TAU, because the tau is only interpretable against it.
        _fam_txt = {"cross": ("GROUPS OF ONE SEED -- ABAG-Rank's group terms, partitioned by seed "
                              "inside group_coeffs; --rank-pairs does not apply"
                              if _group_form else
                              "CROSS-SEED pairs only (different seeds) -- the deployed objective"),
                    "within": "WITHIN-SEED pairs only (same seed) -- a PROXY for selection, not "
                              "the deployed objective",
                    "all": "BOTH families at ONE tau"}['cross']
        print(f"  PAIR FAMILY: {_fam_txt}")
        # NO STOP HERE, AND THE EARLIER ONE WAS WRONG. It refused --rank-pairs within/all
        # without an explicit --rank-tau, on the grounds that at the cross-seed tau a within-seed
        # pair "sits at m/tau ~ 0 where the softplus gives a correct pair and an inverted one the
        # same weight, so the term is inert". The premise is right and the conclusion is not.
        # At overfit30's within-seed median gap of 0.0137 against tau 0.0305 the weights are
        # 0.376 correct / 0.624 inverted -- real discrimination; even at the full training set's
        # 0.0036 they are 0.467 / 0.533, and BOTH PUSH THE SAME WAY, toward the better-DockQ
        # pose. They add, they do not cancel. What a large tau costs is SATURATION: the loss
        # never stops pushing pairs it has already ordered, so the term degenerates toward the
        # linear loss. That is what the --rank-tau help means by the original units error being
        # "inert" -- the softplus NONLINEARITY is inert, not the gradient. Within-seed pairs are
        # under-weighted (6.7% of pairwise gradient mass on overfit30), which is a question for
        # --rank-within-weight, not for tau. So this prints the regime and lets the run proceed.
        if 'cross' != "cross" and a.rank_form == "softplus":
            _r = 0.0137 / _tau_eff          # overfit30 within-seed median; the training pool's
            _sg = 1.0 / (1.0 + math.exp(-_r))
            print(f"  within-seed pairs at tau {_tau_eff:g}: a median-gap pair (0.0137 on "
                  f"overfit30) gets weight {1 - _sg:.3f} if correctly ordered and {_sg:.3f} if "
                  f"inverted ({_sg / (1 - _sg):.2f}x). Re-measure for another pool with "
                  "measure_tau.py --pairs within --restrict-to <pool>/structures.csv; the full "
                  "training set gives 0.0036 and valfull 0.0011, so this ratio is "
                  "POPULATION-SPECIFIC and tau must never be set from val.")
        if a.rank_form == "softplus":
            print(f"  softplus tau {_tau_eff:g}"
                  + ("  (rank_loss.DEFAULT_TAU, the measured median cross-seed |ipTM gap| over "
                     "the full training set)" if _rank_tau is None else "  (given)"))
        elif a.rank_form == "expected":
            # TWO DIFFERENT QUANTITIES SHARE THE --rank-tau FLAG, so say which one is in force.
            # A margin scale of 0.0305 and a softmax temperature of 0.05 are not comparable, and
            # defaulting silently to DEFAULT_TAU here would be a 1.6x sharper selection than
            # intended with nothing in the log to show it.
            _tau_sm = _rank_tau if _rank_tau is not None else _RKL_cfg.DEFAULT_SOFTMAX_TAU
        if a.rank_cross_seed:
            print(f"  CROSS-SEED GROUPS: {a.rank_structs_per_step} structure(s) x all seeds per "
                  f"step; a pose's coefficient uses every other pose of its interface, so pass A "
                  f"scores the whole step before the first backward")
        elif a.rank_form in ("spearman", "distance", "spearman_distance"):
            raise SystemExit(
                f"STOP: --rank-form {a.rank_form} requires --rank-cross-seed. Its coefficients "
                "depend on every score in the group, so they need pass A, and pass A only runs on "
                "the cross-seed step grouping. THIS IS NOT A CROSS-SEED OBJECTIVE: the groups are "
                "partitioned by seed inside group_coeffs, so each group is one seed's --nsample "
                "poses. --rank-cross-seed is what makes a step hold whole structures so pass A "
                "can score them; the grouping it implies is then overridden.")
        elif a.rank_form in ("softplus", "expected"):
            raise SystemExit(
                f"STOP: --rank-form {a.rank_form} without --rank-cross-seed is not implemented. "
                "Its coefficients need pass A, which is wired to the cross-seed step grouping; "
                "running it within-seed would silently fall through to the linear per-unit path "
                "and train a different objective than the flags say. For the expected form this "
                "would also defeat its purpose outright -- the softmax has to run over ALL poses "
                "competing for selection, and within one seed it would range over 5 poses rather "
                "than 50.")
        if _rl_missing:
            raise SystemExit(
                f"STOP: {len(_rl_missing)} scored structure(s) have no DockQ labels in {_rl_p}, "
                f"e.g. {_rl_missing[:6]}. Their poses would pay a forward and contribute no "
                f"ranking gradient, making this a partial experiment.")
        # COVERAGE IS CHECKED AFTER THE MANIFEST IS BUILT, not here: the expected key set is
        # (pdb, seed, pair) over the units this run will actually train, and `want` does not exist
        # yet at this point in main(). An earlier version referenced it here anyway and would have
        # raised NameError at startup the first time --rank-loss was passed. Found in review of
        # a3086ef. See _rank_check_coverage below.

    def _rank_key(pdb, arm_seed_or_int, pair):
        """THE label key, built in exactly one place.

        A unit's identifier is the label/trunk basename -- "free_500000", not 500000 -- because
        that is what names the files. The guard built its keys from the manifest, where seeds are
        plain ints, while the lookup in run_unit did int(arm_seed) on the basename and raised
        ValueError on the first unit. Both sides were internally consistent and disagreed with
        each other, which is the same shape as the reference-vs-model chain bug: two independent
        key constructions. There is now one, so they cannot diverge again.
        """
        sd = arm_seed_or_int
        if isinstance(sd, str):
            sd = sd.rsplit("_", 1)[-1]
        return (pdb, int(sd), pair)

    def _rank_check_coverage(want_units):
        """Every (pdb, seed, interface) this run will train must carry --nsample finite DockQ.

        DRIVEN FROM THE EXPECTED SET, NOT FROM THE LABELS. Two earlier versions were weaker and
        both would have passed the run that was actually broken: one checked only that a structure
        appeared SOMEWHERE in the labels (a pdb-level test cannot see a per-pair hole), and one
        iterated over the labels and skipped keys absent from _scored_asym (so a scored pair with
        no labels at all was invisible). The reference-vs-model chain bug survived both. The
        expected set is built from the manifest and the scored pairs, and set difference does the
        work.
        """
        # NOT UNDER --loss-only. The held-out monitor is a subprocess on the VALIDATION fold
        # (seeds 400000+), it inherits rank_loss=True only so the resume guard matches, and the
        # ranking term never runs there -- a measurement takes no step. Demanding training-fold
        # labels for validation units is a check it cannot satisfy, and it killed the held-out
        # pass on every epoch of s7dh7o: "540 group(s) have NO labels, e.g. ('7qu2', 400000,
        # (0, 2))". FIFTH instance of this class -- --train-only, LoRA, the evaluator's
        # --pose-perblock-share, the ranking guard fields, now a check of my own that a
        # measurement cannot pass.
        # UNCHANGED BY --rank-cross-seed, and that is not an oversight. Cross-seed grouping
        # changes which poses are COMPARED, not which labels must exist: every (pdb, seed,
        # interface) the run touches still needs its --nsample DockQ values, and the groups are
        # built from the same manifest units this set is built from. A group missing a seed's
        # labels is therefore already caught here, before pass A ever runs.
        if not a.rank_loss or a.loss_only:
            return
        expected = {_rank_key(pdb, sd, pr) for pdb, sd in want_units
                    for pr in (_scored_asym.get(pdb) or ())}
        if not expected:
            raise SystemExit("STOP: --rank-loss expected zero (pdb, seed, interface) groups. "
                             "Either the manifest or --pae-scored-pairs selects nothing.")
        missing = sorted(expected - set(_rank_lab))
        short, nonfinite = [], []
        for k in sorted(expected & set(_rank_lab)):
            m = _rank_lab[k]
            if sorted(m) != list(range(a.nsample)):
                short.append((k, sorted(m)))
            if not all(math.isfinite(v) for v in m.values()):
                nonfinite.append(k)
        if missing or short or nonfinite:
            raise SystemExit(
                f"STOP: --rank-loss label coverage is incomplete for the {len(expected)} "
                f"(pdb, seed, interface) group(s) this run trains.\n"
                + (f"  {len(missing)} group(s) have NO labels, e.g. {missing[:6]}\n"
                   if missing else "")
                + (f"  {len(short)} group(s) lack samples 0..{a.nsample - 1}, e.g. {short[:4]}\n"
                   if short else "")
                + (f"  {len(nonfinite)} group(s) carry a non-finite DockQ, e.g. "
                   f"{nonfinite[:4]}\n" if nonfinite else "")
                + "  Those poses would pay a forward and contribute no ranking gradient.")
        if is_main:
            print(f"  ranking label coverage: all {len(expected)} (pdb, seed, interface) group(s) "
                  f"hold {a.nsample} finite DockQ value(s)")

    def _check_eval_pool():
        """Stat the training-eval pool now, not at the first epoch boundary.

        eval_checkpoint shells out to phase_k_eval.py, which hard-stops on a missing trunk or a
        pool interface with no scored pose. That SystemExit lands on RANK 0 ONLY, while ranks 1
        and 2 sit in dist.barrier() until --dist-timeout-min, so a bad pool costs hours of
        training and then a four-hour zombie. The default fold pool really does contain a
        structure the manifest skips (9mx1 has no trunk at all), so this is one wrong path away.
        """
        if not a.train_eval_pool:
            return
        # THE POOL'S OWN FOLD, matching what eval_checkpoint will pass. Checking against --fold
        # when --train-eval-fold overrides it rejected a perfectly good validation pool for
        # having no trunks in the TRAINING fold, which is exactly where it should not have any.
        _f = a.train_eval_fold or a.fold
        if "," in _f:
            raise SystemExit(f"STOP: --train-eval-pool needs a single fold; got {_f!r}. "
                             f"phase_k_eval treats --fold as one directory, so a comma-separated "
                             f"root would be read as a literal path.")
        need = [a.train_eval_pool, os.path.join(_f, "results.csv"),
                os.path.join(_f, "chain_roles.csv")]
        gone = [f for f in need if not os.path.exists(f)]
        if gone:
            raise SystemExit(f"STOP: --train-eval-pool needs {gone}, which do not exist.")
        import csv as _csv2
        # --manifest-skip describes the TRAINING manifest, so it says nothing about a pool in
        # another fold; skipping a structure from training is not a reason it cannot be scored.
        skip = ({x.strip() for x in (a.manifest_skip or "").split(",") if x.strip()}
                if not a.train_eval_fold else set())
        pool_pdbs = {r["pdb"] for r in _csv2.DictReader(open(a.train_eval_pool))}
        bad = sorted(pool_pdbs & skip)
        if bad:
            raise SystemExit(f"STOP: --train-eval-pool names {bad}, which --manifest-skip excludes "
                             f"from training. The evaluator hard-stops on a structure with no "
                             f"trunk, on rank 0, with the other ranks parked at a barrier.")
        notrunk = sorted(p for p in pool_pdbs
                         if not os.path.isdir(os.path.join(_f, "trunks", p)))
        if notrunk:
            raise SystemExit(f"STOP: --train-eval-pool names {len(notrunk)} structure(s) with no "
                             f"trunk directory, e.g. {notrunk[:4]}.")
        print(f"train-eval pool: {len(pool_pdbs)} structure(s), all present, none skipped")

    def _pae_mask_coverage(units):
        """Every structure this run trains on must be named by the pool, checked BEFORE training.

        The running guard below only fires while NOTHING has been masked yet, so once a single
        unit masks successfully an unlimited number of later units can fall back to the UNMASKED
        PAE term in silence. Review reproduced one masked call followed by 100 unmasked ones with
        no error. A per-structure coverage check at startup closes that: a structure missing from
        the pool is a manifest/pool mismatch, which is a setup error, and it is cheap to find here
        rather than impossible to notice later.
        """
        if not _scored_asym:
            return
        want = sorted({t[0] for t in units})
        gone = [p for p in want if not _scored_asym.get(p)]
        if not gone:
            print(f"  PAE mask: all {len(want)} training structure(s) are named by "
                  f"{a.pae_scored_pairs}")
            return
        print(f"  PAE mask: ACCEPTING UNMASKED FALLBACK for {len(gone)} of {len(want)} "
              f"structure(s) (--allow-unmasked-fallback): {gone[:8]}")

    def _pae_mask_check():
        """A mask that never fires is a baseline arm wearing the masked arm's label.

        Both fallbacks below are silent by design -- they keep training rather than dying on one
        odd structure. But if NOTHING is ever masked, the startup banner has announced an objective
        the run is not training. The classic way in is pointing --pae-scored-pairs at the wrong
        pool (the validation one, say): the banner reports 92 structures, every training unit then
        misses, and the run trains the unmasked term under the masked arm's name.
        """
        st = _pae_mask_stats
        if _scored_asym and st["masked"] == 0 and st["unmasked"] >= 20:
            raise SystemExit(
                f"STOP: --pae-scored-pairs matched NO unit in the first {st['unmasked']} seen. "
                f"The pool names {len(_scored_asym)} structure(s), none of which this run trains "
                f"on, so every pose fell back to the UNMASKED PAE term while the run is labelled "
                f"masked. Check that {a.pae_scored_pairs} covers --manifest-selection.")

    def _set_pae_mask(pdb, asym, has_frame, unit=None, pairs_override=None):
        """[N_frame, N_token] over the CROPPED tokens: keep only scored cross-chain entries.

        Falls back to no mask -- never to an all-zero one. An empty mask would make PAELoss divide
        by eps and return a meaningless number rather than fail, which is the worst outcome.

        pairs_override EXISTS FOR --rank-form rel. _scored_asym is built once per structure from
        chain_roles, while rel resolves each interface's asym pair PER POSE through OST. On the 52
        structures where the two disagree, PAE would anchor a different interface from the one rel
        is ranking -- 7stf's (3,1) against (3,0) -- so the two terms would be supervising
        different objects on the same pose. That is objective misalignment, not gradient noise.
        Raised in review. The override is per pose because OST's answer is.
        """
        _pae_keep["pdb"], _pae_keep["unit"] = pdb, unit
        if not _scored_asym and pairs_override is None:
            return
        # COUNTED AND REPORTED, because "the override is wired" and "the override CHANGES
        # something" are different claims and only the second matters. Twice today a change to
        # this code path produced numbers bit-identical to the version it replaced -- once
        # because --ost-mapping was not forwarded to the workers, once here -- and in both cases
        # the identical numbers were the only evidence anything was wrong. So the run says how
        # many poses got an override and on how many it DIFFERED from the static pairs.
        if pairs_override is not None:
            _ov = _pae_keep.setdefault("ov", {"poses": 0, "differed": 0, "pdbs": set()})
            _ov["poses"] += 1
            if set(pairs_override) != set(_scored_asym.get(pdb) or ()):
                _ov["differed"] += 1
                _ov["pdbs"].add(pdb)
        pairs = pairs_override if pairs_override is not None else _scored_asym.get(pdb)
        if not pairs:
            _pae_keep["m"] = None
            _pae_keep["blocks"] = None
            _pae_mask_stats["unmasked"] += 1
            _pae_mask_check()
            return
        aa = asym.reshape(-1).long()
        fa = aa[has_frame.reshape(-1).bool()]
        keep = torch.zeros((fa.numel(), aa.numel()), dtype=torch.bool, device=aa.device)
        # PER-PAIR BLOCKS, KEPT SEPARATELY. Selection reads chain_pair_iptm[x, y] -- one max per
        # INTERFACE, each over just those two chains and with its own d0. 298 of the 368 training
        # structures score two or more pairs, so ranking rows on the union would rank a row in the
        # antigen by its heavy- and light-chain targets averaged together, which is a quantity no
        # ipTM reads. --pae-row-frac weights each block on its own t_i; the union is still what the
        # unweighted term uses, and the blocks sum back to it at uniform weights.
        blocks = []
        for x, y in pairs:
            blk = (((fa[:, None] == x) & (aa[None, :] == y))
                   | ((fa[:, None] == y) & (aa[None, :] == x)))
            keep |= blk
            if bool(blk.any()):
                blocks.append((blk, int(((aa == x) | (aa == y)).sum())))
        if not bool(keep.any()):
            _pae_keep["m"] = None
            _pae_keep["blocks"] = None
            _pae_mask_stats["unmasked"] += 1
            _pae_mask_check()
            return
        _pae_keep["m"] = keep
        _pae_keep["blocks"] = blocks
        _pae_mask_stats["masked"] += 1

    w = {k: loss_fn.loss_weight[k] for k in ("plddt_loss", "pde_loss", "pae_loss", "resolved_loss")}
    print("loss weights: " + "  ".join(
        f"{k.replace('_loss','')} {v:g}" + (" [DROPPED]" if k in _dropped_terms else "")
        for k, v in w.items()))
    _check_eval_pool()
    zero = [k for k, v in w.items() if not v and k not in _dropped_terms]
    # THE GUARD'S PREMISE IS "at 0 the ipTM cannot move", AND A RANKING TERM FALSIFIES IT. This
    # check predates --rank-loss, when PAE was the only path from a weight to chain_pair_iptm. The
    # ranking term differentiates ipTM directly, so alpha_pae 0 with --rank-loss is a RANKING-ONLY
    # arm, which is a deliberate configuration and not the silent no-op the guard was written to
    # catch. Every other zero-weight term is still refused, and pae_loss at 0 WITHOUT a ranking
    # term still is too -- that really would train nothing the project is judged on.
    if zero == ["pae_loss"] and a.rank_loss and float(a.alpha_rank) > 0:
        print(f"  alpha_pae 0 WITH --rank-loss: this is a RANKING-ONLY arm. No PAE gradient is "
              f"taken; chain_pair_iptm moves through the ranking term alone. The held-out monitor "
              f"still measures at alpha_pae 1, so val10 PAE stays comparable to every other arm.")
        zero = []
    if zero:
        raise SystemExit(f"STOP: {', '.join(zero)} carries weight 0, so that term cannot train. "
                         f"FT3 wants alpha_pae = 1 (Protenix's own comment, configs_base.py:402); "
                         f"at 0 the ipTM this project is judged on cannot move. If you meant a "
                         f"ranking-only arm, pass --rank-loss: the ranking term differentiates "
                         f"ipTM directly and the guard allows alpha_pae 0 in that case.")

    # THE RESOLVED DATASET, returned rather than left in locals. main() puts these on the context;
    # the modules downstream read them as rt.<name>. Names follow the live trainer exactly so the
    # equivalence test still lines up.
    return {"loss_fn": loss_fn, "_scored_asym": _scored_asym, "_rank_lab": _rank_lab}
