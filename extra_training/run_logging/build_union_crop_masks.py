#!/usr/bin/env python
"""Precompute the LABEL-FREE union crop as one token mask per structure, for phase_k_eval.

WHY A PRECOMPUTE STEP. The crop is the union over ALL of a structure's candidate poses, which live
across 30 separate seed units. phase_k_eval walks units one at a time, so it cannot build the
union inline -- by the time it sees unit 7 it has not seen unit 8. This writes the union once and
the evaluator loads it.

WHY ONE MASK PER STRUCTURE AND NOT PER POSE. ipTM's d0 is set from the chain pair's token count,
so cropping each pose separately changes the scoring function between the candidates being ranked
(measured: d0 swinging 3.00 -> 4.81 A inside one interface, run_logging/predcrop_size_control.py).
One shared mask gives every pose the same tokens, the same n and the same d0.

NO LABELS. Coordinates come from `ci_coordinate` -- what the fold PREDICTED for each pose. The
union includes the interfaces of poses that dock on the wrong face, so the mask does not encode
where the true epitope is. `--verify-rule` re-runs the same code on the LABEL coordinates and
asserts it reproduces interface_crop_tokens bit-identically, which is the only guarantee that the
rule being applied here is the rule the arm trained under.

OUTPUT: an npz of <pdb> -> bool[n_token], plus a <pdb>__n entry recording how many poses went into
each union, so a short union (a structure with few usable poses) is visible rather than silent.
"""
import argparse
import csv
import os
import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"))
import phase_j_build_cache as BC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True, help="csv with a pdb column")
ap.add_argument("--trunks", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--contact", type=float, default=5.0)
ap.add_argument("--margin", type=float, default=5.0)
ap.add_argument("--seeds", type=int, default=0, help="0 = every unit that HAS A LABEL npz")
ap.add_argument("--seed-base", type=int, default=None,
                help="first seed of the evaluated range. With --budget this names the EXACT "
                     "units the union is built from, e.g. --seed-base 400000 --budget 30 uses "
                     "free_400000..free_400029 and nothing else. Raised in review: the union "
                     "must contain precisely the candidates being evaluated, and relying on "
                     "'whichever units happen to have a label npz' made that true only by "
                     "coincidence -- a label rebuild that added seeds would silently widen "
                     "every crop.")
ap.add_argument("--budget", type=int, default=None,
                help="number of seeds in the evaluated range, matching phase_k_eval --budget.")
ap.add_argument("--nsample", type=int, default=5,
                help="poses per seed, matching the fold. --require-poses is derived as "
                     "--budget * --nsample when both are given, instead of being hardcoded.")
ap.add_argument("--require-poses", type=int, default=0,
                help="hard-stop if any structure EXAMINED fewer candidate poses than this. "
                     "THE UNION MUST SPAN EXACTLY THE POSES THE EVAL SCORES. Here the two "
                     "coincide because the label npzs exist for seeds 400000-400029 and the "
                     "b30 eval scores those same 30 seeds -- but that is a coincidence of how "
                     "the labels were built, not a contract, and a label rebuild that added "
                     "seeds would silently widen every crop. Pass 150 for a b30 pool.")
ap.add_argument("--force-gt", action="store_true",
                help="UNION THE GROUND-TRUTH CONTACT TOKENS IN, for each scored chain pair, on "
                     "top of the pose union. Makes interface coverage 1.0 BY CONSTRUCTION rather "
                     "than 1.0 at the median: measured on TRAIN at margin 2.5 the pose union "
                     "alone gives coverage p10 0.9500, so the bottom decile of structures loses "
                     "5%+ of the very contacts the loss scores -- and because the DockQ label is "
                     "a GLOBAL number computed on the full structure, it cannot be cropped to "
                     "match a partially-kept interface. The head would be asked to score an "
                     "interface that is not in its input. Costs almost nothing: median 0 tokens "
                     "added, worst 6 over the structures measured. REQUIRES --roles and a pool "
                     "with chain1/chain2. This makes the mask NOT label-free, which is fine for a "
                     "TRAINING crop with uncropped inference and wrong for a deployment claim.")
ap.add_argument("--roles", default="", help="chain_roles.csv; required by --force-gt")
ap.add_argument("--ost-pairs", action="store_true",
                help="--force-gt ONLY: also resolve each pose's scored chain pair the way the "
                     "TRAINER does -- per pose, through OST -- and force those pairs' contacts in "
                     "too. WHY IT IS NOT OPTIONAL IN PRACTICE: phase_k_train._pose_pairs resolves "
                     "the pair per pose because chain_roles.csv assigns one reference chain per "
                     "structure and that is unanswerable when two reference chains are chemically "
                     "identical. Its own docstring records that the static and pose-resolved "
                     "mappings disagree on 1,095 of 18,400 (structure, seed, pose) checks across "
                     "52 of 368 structures, and that 7stf resolves to a DIFFERENT CHAIN rather "
                     "than a reordering. So a forced set built from the static pairs alone "
                     "guarantees coverage of a pair the trainer may not be supervising, which is "
                     "the hole --force-gt exists to close. Forcing the UNION of both mappings "
                     "covers whichever one the trainer picks. Needs --fold.")
ap.add_argument("--fold", default="", help="fold root, for the OST unit dirs; needs --ost-pairs")
ap.add_argument("--verify-rule", action="store_true", default=True)
a = ap.parse_args()
if a.force_gt and not a.roles:
    raise SystemExit("STOP: --force-gt needs --roles to map the pool's reference chains onto the "
                     "model chain indices the cached features use.")
if a.ost_pairs and not a.force_gt:
    raise SystemExit("STOP: --ost-pairs only affects the forced ground-truth set, so it does "
                     "nothing without --force-gt.")
if a.ost_pairs and not a.fold:
    raise SystemExit("STOP: --ost-pairs needs --fold to find each pose's OST unit directory.")


def crop_from_coords(z, coords, cmask_all, rep, contact, margin):
    """interface_crop_tokens' rule over an arbitrary (n_pose, n_atom, 3) coordinate array."""
    asym = np.asarray(z["ci_feat_asym_id"]).reshape(-1)
    a2t = np.asarray(z["ci_feat_atom_to_token_idx"]).reshape(-1)
    n_tok = asym.shape[0]
    in_range = (a2t >= 0) & (a2t < n_tok)
    keep = np.zeros(n_tok, dtype=bool)
    used = 0
    seen = 0
    for pi in range(coords.shape[0]):
        seen += 1
        cmask = cmask_all[pi].astype(bool) & in_range
        if cmask.sum() < 4:
            continue
        xyz, tok_of_atom = coords[pi][cmask], a2t[cmask]
        chain_of_atom = asym[tok_of_atom]
        if np.unique(chain_of_atom).size < 2:
            continue
        iface = np.zeros(n_tok, dtype=bool)
        for i, j in cKDTree(xyz).query_pairs(contact):
            if chain_of_atom[i] != chain_of_atom[j]:
                iface[tok_of_atom[i]] = True
                iface[tok_of_atom[j]] = True
        if not iface.any():
            continue
        tok_xyz = np.full((n_tok, 3), np.nan, dtype=np.float64)
        for ai in np.nonzero(rep & cmask)[0]:
            t = int(a2t[ai])
            if np.isnan(tok_xyz[t, 0]):
                tok_xyz[t] = coords[pi][ai]
        good = ~np.isnan(tok_xyz[:, 0])
        centres = np.nonzero(iface & good)[0]
        if centres.size == 0:
            keep |= iface
            used += 1
            continue
        gi = np.nonzero(good)[0]
        d = np.linalg.norm(tok_xyz[gi][:, None, :] - tok_xyz[centres][None, :, :], axis=-1)
        near = np.zeros(n_tok, dtype=bool)
        near[gi[(d < margin).any(axis=1)]] = True
        keep |= iface | near
        used += 1
    return ((keep, used, seen) if used and keep.any() else (None, used, seen))


pdbs = []
for r in csv.DictReader(open(a.pool)):
    if r["pdb"] not in pdbs:
        pdbs.append(r["pdb"])

def gt_contact_tokens(z, lab, pairs, contact):
    """Bare ground-truth contact tokens for the SCORED chain pairs, from label coordinates.

    No margin: this set exists only to guarantee the loss's own tokens survive the crop. Returns
    None when no scored pair resolves, so the caller skips the structure rather than writing a
    mask that silently guarantees nothing.
    """
    asym = np.asarray(z["ci_feat_asym_id"]).reshape(-1)
    a2t = np.asarray(z["ci_feat_atom_to_token_idx"]).reshape(-1)
    n_tok = asym.shape[0]
    lc = np.asarray(lab["label_coordinate"])[0]
    lm = np.asarray(lab["label_coordinate_mask"])[0].astype(bool)
    ok = lm & (a2t >= 0) & (a2t < n_tok)
    ch = asym[a2t]
    ti = np.zeros(n_tok, dtype=bool)
    any_pair = False
    for (x, y) in sorted(pairs):
        m1, m2 = ok & (ch == x), ok & (ch == y)
        if not m1.any() or not m2.any():
            continue
        any_pair = True
        i1, i2 = np.where(m1)[0], np.where(m2)[0]
        for q, h in enumerate(cKDTree(lc[m2]).query_ball_point(lc[m1], contact)):
            if h:
                ti[a2t[i1[q]]] = True
                for j in h:
                    ti[a2t[i2[j]]] = True
    return ti if (any_pair and ti.any()) else None


def ost_pairs_for(pdb, labels_root, fold, refs, seeds_wanted, nsample):
    """Every (ax, ay) the TRAINER could supervise for this structure, resolved per pose via OST.

    Returns (pairs, n_checked, n_new_vs_static) or (None, 0, 0) when OST cannot be read for any
    pose -- the caller then falls back to the static pairs and SAYS SO, because silently covering
    only the static mapping is the defect this exists to remove.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"))
    import phase_k_eval as _PKE
    import phase_j_build_cache as _BCl
    got_pairs, n_ok = set(), 0
    for sd in seeds_wanted:
        try:
            ud = _BCl.seed_dir(fold, pdb, int(sd))
        except Exception:                                            # noqa: BLE001
            continue
        for k in range(nsample):
            pr, _why = _PKE.ost_chain_pairs(ud, pdb, int(k), refs)
            if pr is None:
                continue
            n_ok += 1
            for _iface, ax, ay in pr:
                got_pairs.add((min(ax, ay), max(ax, ay)))
    return (got_pairs or None), n_ok, 0


# THE SCORED PAIRS, as chain INDICES into asym_id. Only needed by --force-gt.
SCORED = {}
if a.force_gt:
    _roles = {}
    for _r in csv.DictReader(open(a.roles)):
        _roles.setdefault(_r["pdb"], {})[_r["ref_chain"]] = _r["model_chain"]
    for _r in csv.DictReader(open(a.pool)):
        _m1 = _roles.get(_r["pdb"], {}).get(_r.get("chain1", ""))
        _m2 = _roles.get(_r["pdb"], {}).get(_r.get("chain2", ""))
        if _m1 and _m2:
            SCORED.setdefault(_r["pdb"], set()).add((min(ord(_m1), ord(_m2)) - ord("A"),
                                                     max(ord(_m1), ord(_m2)) - ord("A")))
    REFS = {}
    if a.ost_pairs:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                        "phases"))
        import phase_k_eval as _PKE0
        REFS = _PKE0.pool_ref_pairs(a.pool)
    if not SCORED:
        raise SystemExit(f"STOP: --force-gt resolved 0 scored pair(s) from {a.pool} via "
                         f"{a.roles}. The pool needs chain1/chain2 columns and the roles csv "
                         f"needs matching pdb/ref_chain rows.")
    print(f"  --force-gt: {sum(len(v) for v in SCORED.values())} scored pair(s) over "
          f"{len(SCORED)} structure(s)")

out, verified, vfail, skipped = {}, 0, [], []
ost_extra, ost_unresolved = {}, []
_seeds_for_ost = (range(a.seed_base, a.seed_base + a.budget)
                  if (a.ost_pairs and a.seed_base is not None and a.budget) else [])
n_nolabel = 0
units_used = {}
for pdb in pdbs:
    tdir, ldir = os.path.join(a.trunks, pdb), os.path.join(a.labels, pdb)
    if not (os.path.isdir(tdir) and os.path.isdir(ldir)):
        skipped.append((pdb, "no trunk/label dir"))
        continue
    if a.seed_base is not None and a.budget:
        # NAMED, NOT DISCOVERED. Every unit in the range must exist; a missing one is fatal
        # rather than a quietly smaller union.
        units = [f"free_{sd}.npz" for sd in range(a.seed_base, a.seed_base + a.budget)]
        miss = [u for u in units if not os.path.exists(os.path.join(tdir, u))]
        if miss:
            raise SystemExit(f"STOP: {pdb} is missing {len(miss)} unit(s) of the evaluated seed "
                             f"range {a.seed_base}..{a.seed_base + a.budget - 1}, e.g. "
                             f"{miss[:3]}. The union would cover fewer candidates than the eval "
                             f"scores.")
    else:
        units = sorted(f for f in os.listdir(tdir) if f.endswith(".npz"))
        if a.seeds:
            units = units[:a.seeds]
    if not units:
        skipped.append((pdb, "no units"))
        continue
    uni, npose, checked = None, 0, False
    nseen, nocontact = 0, 0
    used_units = []
    for u in units:
        try:
            z = np.load(os.path.join(tdir, u), allow_pickle=True)
            lab = np.load(os.path.join(ldir, u), allow_pickle=True)
        except FileNotFoundError:
            # EXPECTED AND NOT AN ERROR: the trunk cache holds 100 seeds per structure and the
            # labels only the 30 the eval scores. Counted, not listed -- 3,710 identical lines
            # of "FileNotFoundError" buried the one skip that would have mattered.
            n_nolabel += 1
            continue
        except (OSError, ValueError) as e:
            skipped.append((pdb, f"{u}: {type(e).__name__}"))
            continue
        rep = np.asarray(lab["lf_pae_rep_atom_mask"]).reshape(-1).astype(bool)
        if a.verify_rule and not checked:
            mine, _, _ = crop_from_coords(z, np.asarray(lab["label_coordinate"]),
                                          np.asarray(lab["label_coordinate_mask"]), rep,
                                          a.contact, 10.0)
            ref, _ = BC.interface_crop_tokens(z, lab, a.contact, 10.0, all_poses=True)
            if ref is not None:
                (vfail.append(pdb) if (mine is None or not np.array_equal(mine, ref))
                 else None)
                verified += int(mine is not None and np.array_equal(mine, ref))
            checked = True
        used_units.append(u)
        pc = np.asarray(z["ci_coordinate"])
        pm = np.isfinite(pc).all(axis=-1)
        k, used, seen = crop_from_coords(z, pc, pm, rep, a.contact, a.margin)
        nseen += seen
        nocontact += seen - used
        if k is not None:
            uni = k if uni is None else (uni | k)
            npose += used
    if uni is None:
        skipped.append((pdb, "no usable pose"))
        continue
    units_used[pdb] = used_units
    n_forced = 0
    if a.force_gt:
        # THE GT CONTACT SET, BARE -- no margin on it. Its only job is to guarantee that every
        # token the loss scores survives; the margin's job is context, and the pose union already
        # provides that. Taken from the LABEL coordinates of pose 0, scored pairs only.
        _pairs_use = set(SCORED.get(pdb, set()))
        _ost_note = "static"
        if a.ost_pairs:
            _op, _nok, _ = ost_pairs_for(pdb, a.labels, a.fold, REFS.get(pdb) or [],
                                         _seeds_for_ost, a.nsample)
            if _op is None:
                # NAMED, NOT SWALLOWED. A structure whose OST cannot be read falls back to the
                # static pairs, and that is exactly the case the audit must re-examine.
                ost_unresolved.append(pdb)
            else:
                _new = _op - _pairs_use
                _pairs_use |= _op
                if _new:
                    ost_extra[pdb] = sorted(_new)
                _ost_note = f"static+ost({_nok} pose(s))"
        gt_i = gt_contact_tokens(z, lab, _pairs_use, a.contact)
        if gt_i is None:
            skipped.append((pdb, "--force-gt: no scored pair resolvable"))
            continue
        n_forced = int((gt_i & ~uni).sum())
        uni = uni | gt_i
        cov_after = float((gt_i & uni).sum() / max(int(gt_i.sum()), 1))
        if cov_after < 1.0:
            raise SystemExit(f"STOP: {pdb} still has coverage {cov_after:.4f} after forcing, "
                             f"which is arithmetically impossible -- the union was not applied.")
    out[pdb] = uni
    out[pdb + "__n"] = np.array([npose, int(uni.sum()), int(uni.size), nseen, nocontact,
                                 n_forced], dtype=np.int64)

# DERIVED, NOT HARDCODED. Raised in review: 150 was this pool's budget x nsample written out by
# hand, which would be silently wrong on any other pool.
_req = a.require_poses or ((a.budget * a.nsample) if (a.budget and a.nsample) else 0)
short = [(k.replace("__n", ""), int(out[k][3])) for k in out
         if k.endswith("__n") and _req and out[k][3] < _req]
if short:
    raise SystemExit(f"STOP: {len(short)} structure(s) EXAMINED fewer than {_req} "
                     f"candidate poses, e.g. {short[:4]}. A union built over a subset of the "
                     f"scored candidates is not the crop this experiment specifies. (Poses that "
                     f"were examined but had no cross-chain contact are fine and counted "
                     f"separately -- they contribute nothing by definition.)")
n_src = len([k for k in out if not k.endswith("__n")])
if vfail:
    raise SystemExit(f"STOP: the crop rule does NOT reproduce interface_crop_tokens on "
                     f"{len(vfail)} structure(s), e.g. {vfail[:4]}. Every mask this would write "
                     f"is a different crop from the one the arm trained under.")
if not out:
    raise SystemExit("STOP: no mask built")

import hashlib
import json
import subprocess
try:
    _rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=os.path.dirname(os.path.abspath(__file__)),
                          timeout=10).stdout.strip() or "unknown"
except Exception:
    _rev = "unknown"
_meta = dict(margin=a.margin, contact=a.contact, pool=a.pool, trunks=a.trunks, labels=a.labels,
             seeds=a.seeds, seed_base=a.seed_base, budget=a.budget,
             nsample=a.nsample, require_poses=_req, n_structures=n_src,
             unit_digest=hashlib.sha256(
                 "|".join(f"{k}:{','.join(units_used[k])}" for k in sorted(units_used))
                 .encode()).hexdigest()[:16],
             builder_git=_rev, built_utc=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                     __import__("time").gmtime()))
out["__meta"] = np.array([json.dumps(_meta, sort_keys=True)])
os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
np.savez_compressed(a.out, **out)
n_s = len([k for k in out if not k.endswith("__n") and k != "__meta"])
_keys = [k for k in out if not k.endswith("__n") and k != "__meta"]
fr = [out[k + "__n"][1] / out[k + "__n"][2] for k in _keys]
np_ = [out[k + "__n"][0] for k in _keys]
print(f"{n_s} structure(s) -> {a.out}")
print(f"  provenance: margin {a.margin} A, contact {a.contact} A, "
      f"unit digest {_meta['unit_digest']}, builder {_meta['builder_git'][:8]}")
print(f"  rule verified against interface_crop_tokens on {verified} structure(s)")
print(f"  contact {a.contact} A, margin {a.margin} A, poses per union: median "
      f"{np.median(np_):.0f}, min {min(np_)}")
print(f"  kept fraction: median {np.median(fr):.4f}, p10 {np.percentile(fr, 10):.4f}, "
      f"max {max(fr):.4f}")
_nc = sum(int(out[k][4]) for k in out if k.endswith("__n"))
_sn = sum(int(out[k][3]) for k in out if k.endswith("__n"))
print(f"  candidate poses examined {_sn}; of those {_nc} had NO cross-chain contact under "
      f"{a.contact} A and contributed nothing (fully dissociated decoys)")
print(f"  units without a label npz (expected, not scored): {n_nolabel}")
if a.ost_pairs:
    print(f"  --ost-pairs: {len(ost_extra)} structure(s) had an OST-resolved pair the static "
          f"pool mapping does NOT name, so their forced set would have protected the wrong "
          f"chain pair without this flag"
          + (f": {sorted(ost_extra)[:6]}" if ost_extra else ""))
    if ost_unresolved:
        print(f"  --ost-pairs: {len(ost_unresolved)} structure(s) had NO readable OST mapping "
              f"and fell back to the static pairs: {ost_unresolved[:6]}. Their coverage is "
              f"NOT guaranteed; the audit must re-examine them.")
if skipped:
    print(f"  SKIPPED {len(skipped)}: {skipped[:5]}")
