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
ap.add_argument("--require-poses", type=int, default=0,
                help="hard-stop if any structure EXAMINED fewer candidate poses than this. "
                     "THE UNION MUST SPAN EXACTLY THE POSES THE EVAL SCORES. Here the two "
                     "coincide because the label npzs exist for seeds 400000-400029 and the "
                     "b30 eval scores those same 30 seeds -- but that is a coincidence of how "
                     "the labels were built, not a contract, and a label rebuild that added "
                     "seeds would silently widen every crop. Pass 150 for a b30 pool.")
ap.add_argument("--verify-rule", action="store_true", default=True)
a = ap.parse_args()


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

out, verified, vfail, skipped = {}, 0, [], []
n_nolabel = 0
units_used = {}
for pdb in pdbs:
    tdir, ldir = os.path.join(a.trunks, pdb), os.path.join(a.labels, pdb)
    if not (os.path.isdir(tdir) and os.path.isdir(ldir)):
        skipped.append((pdb, "no trunk/label dir"))
        continue
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
    out[pdb] = uni
    out[pdb + "__n"] = np.array([npose, int(uni.sum()), int(uni.size), nseen, nocontact],
                                dtype=np.int64)

short = [(k.replace("__n", ""), int(out[k][3])) for k in out
         if k.endswith("__n") and a.require_poses and out[k][3] < a.require_poses]
if short:
    raise SystemExit(f"STOP: {len(short)} structure(s) EXAMINED fewer than {a.require_poses} "
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
             seeds=a.seeds, require_poses=a.require_poses, n_structures=n_src,
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
if skipped:
    print(f"  SKIPPED {len(skipped)}: {skipped[:5]}")
