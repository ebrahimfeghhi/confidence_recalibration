#!/usr/bin/env python
"""Does Protenix's interface crop destroy the iTM label's ability to rank poses?

THE QUESTION. Cropping is attractive for speed -- pair ops are O(N^2) and the measured cost
exponent on this arm is 2.0-2.5 -- but the relative-iTM loss trains the head to reproduce the TRUE
iTM of each pose, and the true iTM is an average over the interface's cross-chain pairs. Crop the
structure and that average is taken over a SUBSET. If the subset still orders a structure's poses
the way the full one does, cropping costs nothing but compute. If it does not, cropping destroys
the label and no speedup is worth it.

SO THIS MEASURES THE LABEL, NOT A MODEL. For every training interface it computes

    rho_full    = Spearman( true iTM over ALL cross-chain pairs , DockQ ) over that interface's poses
    rho_crop(c) = Spearman( true iTM over the pairs crop c KEEPS  , DockQ ) over the same poses

and reports mean_c rho_crop(c) - rho_full. Negative means the crop costs ranking signal.

PROTENIX'S CROP IS USED UNCHANGED. protenix.utils.cropping.get_spatial_crop_index with
interface_crop=True is called directly -- the same function the shipped training pipeline uses,
with the same random interface anchor and the same k-nearest-to-anchor token set. Nothing about it
is tuned here, because the point is to characterise it as it is.

ONE CROP PER STRUCTURE, APPLIED TO EVERY POSE. A crop that differed per pose would make the iTM
gap between two poses measure the crop rather than the poses, which is the whole reason the union
crop exists. Each sampled crop is therefore applied to all of a structure's poses before any
correlation is taken.

d0 IS PINNED AT THE UNCROPPED TOKEN COUNT, at the user's direction. d0 = 1.24(N-15)^(1/3) - 1.8
is TM-score's length normalisation, and shrinking it with the crop would make the kernel steeper
-- scoring the same error worse -- for a task cropping does not make easier. Pinning it also
means cropped and uncropped iTM differ by exactly ONE thing, which pairs are in the average,
rather than by two. --d0-from-crop switches to Protenix's natural behaviour for comparison.

GT INTERFACE RETENTION IS REPORTED ALONGSIDE, because the expected failure mode is a crop whose
random anchor lands away from the antibody-antigen interface -- the contact test inside
get_interface_token accepts ANY pair of distinct chains, heavy-light packing included, so an
anchor can centre on the wrong interface entirely. retention is the fraction of the interface's
own cross-chain contact tokens that survive the crop.

Usage:
    python -I run_logging/crop_itm_rank.py --crops 1000 --workers 24 \
        --out training/runs/crop_itm_rank.csv
    python -I run_logging/crop_itm_rank.py --plot training/runs/crop_itm_rank.csv \
        --fig figures/crop_itm_rank.png
"""
import argparse
import binascii
import collections
import csv
import json
import os
import sys
import time

# THREAD CAPS GO BEFORE numpy, because OpenBLAS reads them at import and ignores them after.
# Uncapped, each shard's sgemm grabs every core it can see: 16 shards on a 288-core box drove the
# load average to 110 while a training run was using the same CPUs for trunk reads and prefetch.
# The budget is a hard ceiling the user set (64 cores), so the default is deliberately modest and
# the launcher multiplies it by the shard count rather than hoping.
# setdefault WAS NOT ENOUGH: it leaves a larger value already exported by the parent shell in
# place, so the cap was advisory rather than enforced. Raised in external review. Assigned
# unconditionally now, and main() verifies the pools BLAS actually built rather than trusting it.
_CT = os.environ.get("CROP_THREADS", "3")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _CT

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run_logging"))
sys.path.insert(0, os.path.join(ROOT, "phases"))

import pae_row_weight as rw                                        # noqa: E402
from phase_k_eval import interface_chain_pairs                     # noqa: E402
from protenix.model.loss import PAELoss                            # noqa: E402
from protenix.utils.cropping import get_spatial_crop_index         # noqa: E402

N_BINS = 64
MIN_BIN, MAX_BIN = 0.0, 32.0
# TRUNK COVERAGE IS NOT UNIFORM, and it decides how many poses an interface can contribute.
# Measured on the current fold: 437 structures have 15 cached trunk seeds, 3 have 10, and 416 have
# only ONE -- the trunks are ~205 MB each and the disk is a hard quota. A rank correlation over 5
# poses is degenerate (it comes out at +-1 most of the time), so --min-poses excludes those rather
# than letting them dominate the histogram with noise.
SEEDS = list(range(500000, 500015))


# ----------------------------------------------------------------------------- data

def unit_tensors(fold, pdb, seed):
    """(asym, has_frame, frame_atom_index, rep_atom_mask, trunk, label) or None. From itm_ceiling."""
    tp = os.path.join(ROOT, fold, "trunks", pdb, f"free_{seed}.npz")
    lp = os.path.join(ROOT, fold, "labels", pdb, f"free_{seed}.npz")
    if not (os.path.exists(tp) and os.path.exists(lp)):
        return None
    tr = np.load(tp, allow_pickle=True)
    lb = np.load(lp, allow_pickle=True)
    return (torch.as_tensor(tr["ci_feat_asym_id"]).long(),
            torch.as_tensor(lb["lf_has_frame"]).bool(),
            torch.as_tensor(lb["lf_frame_atom_index"]).long(),
            torch.as_tensor(lb["lf_pae_rep_atom_mask"]).bool(),
            tr, lb)


def pose_bins(pl, tr, lb, p, hf, fai, rep):
    """(bin index [N,N] int64, pair mask [N_frame, N] bool) for one pose. From itm_ceiling."""
    _, bins, pm = pl.calculate_label(
        pred_coordinate=torch.as_tensor(tr["ci_coordinate"][p]).float().unsqueeze(0),
        true_coordinate=torch.as_tensor(lb["label_coordinate"][p]).float(),
        coordinate_mask=torch.as_tensor(lb["label_coordinate_mask"][p]).float(),
        rep_atom_mask=rep, frame_atom_index=fai, has_frame=hf)
    return bins[0].argmax(-1).contiguous(), pm


def token_geometry(lb, rep):
    """(token xyz [N,3] float32, resolved mask [N] bool) from the TRUE structure of pose 0.

    Protenix crops the ground-truth bioassembly, so the distances that define the crop come from
    true coordinates. Representative atoms are the token positions.
    """
    xyz = torch.as_tensor(lb["label_coordinate"][0]).float()
    msk = torch.as_tensor(lb["label_coordinate_mask"][0]).bool()
    return xyz[rep], msk[rep]


# ----------------------------------------------------------------------------- iTM

def weight_vector(nd, cache):
    wb = cache.get(nd)
    if wb is None:
        wb = rw._tm_weights(N_BINS, rw.calculate_normalization(nd),
                            torch.device("cpu"), torch.float32, MIN_BIN, MAX_BIN)
        cache[nd] = wb
    return wb


def itm_over_subsets(W, M, keep, row_tokens):
    """Mean-pooled iTM for every pose under every crop, for ONE interface.

    W          [P, F, N] float32 -- TM weight of each (frame row, token) pair's TRUE bin
    M          [F, N]    float32 -- 1 where the pair is a scored cross-chain pair of this interface
    keep       [N, C]    float32 -- column c is the indicator of crop c's kept tokens
    row_tokens [F]       int     -- the TOKEN index each frame row belongs to

    Returns [P, C] and the per-crop count of rows that survived.

    A ROW IS ONLY POOLED IF ITS OWN TOKEN SURVIVES THE CROP. The first version masked only the
    target COLUMNS and then pooled any row with a surviving partner -- so a row whose anchor token
    had been cropped away still contributed, which inflates the overlap between cropped and
    uncropped iTM and therefore biases the whole analysis toward "cropping is harmless". Caught in
    external review; reproduced against a slow sliced reference at 0.078 absolute on a 12-token
    fixture before the fix, and 0 after. phases.calculate_iptm pools over tokens PRESENT in the
    input, which is what row_keep reproduces.

    WHY A MATMUL. Naively this is C x P x F x N work -- 1000 crops x 50 poses x 600 x 600 is
    1.8e10 operations per interface and would take days over the training set. The row sum
    sum_j W[p,i,j] M[i,j] 1[j in c] is linear in the crop indicator, so all C crops are one
    sgemm: (W*M) @ keep.
    """
    P, F, N = W.shape
    WM = (W * M[None]).reshape(P * F, N)
    num = (WM @ keep).reshape(P, F, -1)             # [P, F, C] sum over kept columns
    den = (M @ keep)                                # [F, C]    how many kept columns per row
    t = num / np.clip(den, 1.0, None)[None]         # [P, F, C] per-row t
    row_keep = keep[row_tokens]                     # [F, C] is this row's own token in the crop
    live = ((den > 0) & (row_keep > 0)).astype(np.float32)
    n_live = live.sum(0)                            # [C]
    return (t * live[None]).sum(1) / np.clip(n_live, 1.0, None)[None], n_live


def spearman(x, y):
    """Spearman rho with average ranks; nan when either side is constant."""
    if len(x) < 3:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def _rank(v):
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v), dtype=np.float64)
    r[order] = np.arange(len(v), dtype=np.float64)
    # average ties, so a plateau of equal DockQ does not get an arbitrary order
    uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    return r


# ----------------------------------------------------------------------------- crops

def sample_crops(xyz, resolved, chain_id, ref_asym, crop_size, n_crops, rng_seed):
    """n_crops token-index arrays from Protenix's SpatialInterfaceCropping, unchanged.

    get_spatial_crop_index wants the distance matrix from the REFERENCE tokens to every token, so
    the rows are restricted to the two chains of this interface exactly as _map_ref_chain does.
    """
    torch.manual_seed(rng_seed)
    N = xyz.shape[0]
    ref_rows = torch.nonzero(
        (chain_id == ref_asym[0]) | (chain_id == ref_asym[1]), as_tuple=True)[0]
    # token_distance IS [REFERENCE ROWS, ALL TOKENS], not [N, N]. get_interface_token indexes its
    # rows by expand_reference_chain_id.nonzero(), i.e. positions WITHIN the reference chains, and
    # a full-length matrix makes that index out of range -- caught by a shape error on the first
    # structure. The returned crop indices are still full-array token indices, because the topk
    # runs over the chosen row's distances to every token.
    d = torch.cdist(xyz[ref_rows], xyz)                            # [R, N]
    m = (resolved[ref_rows].unsqueeze(1) & resolved.unsqueeze(0)).float()
    out = []
    for _ in range(n_crops):
        idx, _anchor = get_spatial_crop_index(
            tokens=torch.arange(N), chain_id=chain_id,
            token_distance=d, token_distance_mask=m,
            reference_chain_id=torch.as_tensor(ref_asym),
            ref_space_uid_token=torch.arange(N),
            crop_size=crop_size, crop_complete_ligand_unstdRes=False,
            interface_crop=True, interface_minimal_distance=15)
        out.append(idx.numpy())
    return out


def crop_seed(base, iface):
    """A per-interface seed that is the SAME on every invocation.

    This was `base + hash(iface) % 10**6`. str.__hash__ is salted per process in Python 3, so two
    runs of this script drew DIFFERENT crops while the docstring promised reproducibility -- the
    kind of defect that only shows up when someone tries to reproduce a number months later.
    crc32 is stable across processes, versions and machines.
    """
    return int(base) + int(binascii.crc32(iface.encode())) % 10 ** 6


# ----------------------------------------------------------------------------- driver

def run_interface(pdb, iface, ax, ay, poses, dockq, xyz, resolved, asym, hf,
                  crop_size, n_crops, pin_d0, rng_seed):
    """One row of the output. poses: list of (bin_idx [N,N], pair_mask [F,N])."""
    N = asym.numel()
    tok_in = (asym == ax) | (asym == ay)
    n_pair_tokens = int(tok_in.sum())
    if n_pair_tokens < 4:
        return None
    hf_rows = torch.nonzero(hf, as_tuple=True)[0]

    # the scored cross-chain pair mask of THIS interface, same rule as itm_ceiling.pair_mask_for
    # ONE MASK FOR EVERY POSE, ASSERTED RATHER THAN ASSUMED. The scored-pair mask comes from
    # pose 0 and is reused, which is only sound if every pose of the structure has the same one.
    # A silent mismatch would change both the label and the regret, so it is checked instead of
    # trusted -- raised in external review.
    pm0 = poses[0][1]
    n_mismatch = sum(1 for p in poses[1:] if not torch.equal(p[1], pm0))
    if n_mismatch:
        return {"pdb": pdb, "iface": iface, "skip_reason": f"pair mask varies across "
                f"{n_mismatch} of {len(poses)} poses"}
    M = (pm0 & (tok_in[hf].unsqueeze(1) & tok_in.unsqueeze(0))
         & (asym[hf].unsqueeze(1) != asym.unsqueeze(0))).float().numpy()
    if M.sum() == 0:
        return None

    wcache = {}
    wb_fixed = weight_vector(n_pair_tokens, wcache).numpy()
    W = np.stack([wb_fixed[p[0][hf].numpy()] for p in poses])   # [P, F, N]

    crops = sample_crops(xyz, resolved, asym, (ax, ay), crop_size, n_crops, rng_seed)
    keep = np.zeros((N, len(crops)), dtype=np.float32)
    for c, idx in enumerate(crops):
        keep[idx, c] = 1.0

    # GT interface retention: of the tokens that actually carry a scored cross-chain pair, how
    # many survive? Rows and columns both matter, so take the union of the two.
    invol = np.zeros(N, dtype=bool)
    invol[np.asarray(hf_rows)[M.any(1)]] = True
    invol |= M.any(0) > 0
    ret = keep[invol].mean(0) if invol.any() else np.zeros(len(crops), dtype=np.float32)

    row_tokens = np.asarray(hf_rows)                # frame row -> token index
    full_keep = np.ones((N, 1), dtype=np.float32)
    itm_full = itm_over_subsets(W, M, full_keep, row_tokens)[0][:, 0]
    rho_full = spearman(itm_full, dockq)

    # REGRET IS THE DECISION-RELEVANT NUMBER, not rho. Spearman grades the whole ordering, but we
    # only ever deploy the argmax, and this project has now watched rho improve four times without
    # selected DockQ following. regret = oracle - DockQ(pose the label picks), so 0 means the label
    # selects the best available pose and larger is worse.
    oracle = float(dockq.max())
    regret_full = oracle - float(dockq[int(np.argmax(itm_full))])

    rhos = np.empty(len(crops), dtype=np.float64)
    regrets = np.full(len(crops), np.nan, dtype=np.float64)
    live_rows = np.empty(len(crops), dtype=np.float64)
    CH = 250
    for lo in range(0, len(crops), CH):
        blk, nlive = itm_over_subsets(W, M, keep[:, lo:lo + CH], row_tokens)   # [P, c], [c]
        live_rows[lo:lo + blk.shape[1]] = nlive
        # A CROP WITH NO SURVIVING SCORED ROW HAS NO iTM AT ALL. Clamping the denominator makes it
        # come out as a confident 0.0, which would then win or lose the argmax on an artefact, so
        # those crops are excluded from both aggregates and counted instead.
        good = nlive > 0
        pick = np.argmax(blk, axis=0)
        regrets[lo:lo + blk.shape[1]][good] = oracle - dockq[pick[good]]
        for j in range(blk.shape[1]):
            rhos[lo + j] = spearman(blk[:, j], dockq) if good[j] else np.nan
    ok = ~np.isnan(rhos) & ~np.isnan(regrets)
    if not ok.any():
        return None
    return {"pdb": pdb, "iface": iface, "n_poses": len(dockq), "n_tokens": N,
            "n_pair_tokens": n_pair_tokens, "crop_size": crop_size,
            "pin_d0": int(pin_d0), "n_crops": len(crops), "n_crops_valid": int(ok.sum()),
            "rho_full": round(rho_full, 6),
            "rho_crop_mean": round(float(rhos[ok].mean()), 6) if ok.any() else float("nan"),
            "rho_crop_std": round(float(rhos[ok].std()), 6) if ok.any() else float("nan"),
            "rho_crop_p10": round(float(np.percentile(rhos[ok], 10)), 6) if ok.any() else float("nan"),
            "rho_crop_p90": round(float(np.percentile(rhos[ok], 90)), 6) if ok.any() else float("nan"),
            "delta": round(float(rhos[ok].mean() - rho_full), 6) if ok.any() else float("nan"),
            "oracle_dockq": round(oracle, 6),
            "regret_full": round(regret_full, 6),
            "regret_crop_mean": round(float(regrets[ok].mean()), 6),
            "regret_crop_std": round(float(regrets[ok].std()), 6),
            "regret_crop_p90": round(float(np.percentile(regrets[ok], 90)), 6),
            "regret_crop_worse_frac": round(float((regrets[ok] > regret_full + 1e-9).mean()), 6),
            "regret_delta": round(float(regrets[ok].mean() - regret_full), 6),
            "n_crops_degenerate": int((live_rows == 0).sum()),
            "live_rows_mean": round(float(live_rows.mean()), 2),
            "retention_mean": round(float(ret.mean()), 6),
            "retention_p10": round(float(np.percentile(ret, 10)), 6),
            "kept_tokens_mean": round(float(keep.sum(0).mean()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="training/fold")
    ap.add_argument("--pool", default="training/combined/pool.csv")
    ap.add_argument("--roles", default="training/fold/chain_roles_all.csv")
    ap.add_argument("--results", default="training/fold/results.csv")
    ap.add_argument("--manifest", default="train_val_splits/cache_train_structures_extended.csv")
    ap.add_argument("--skip", default="4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m")
    ap.add_argument("--crops", type=int, default=1000)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--d0-from-crop", action="store_true",
                    help="use Protenix's natural d0 (from the cropped token count) instead of "
                         "pinning it at the uncropped count")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-poses", type=int, default=25,
                    help="skip an interface with fewer scored poses than this; a Spearman over 5 "
                         "poses is degenerate and 416 of 846 structures have only one trunk seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="training/runs/crop_itm_rank.csv")
    a = ap.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(int(_CT))
    # VERIFY, DO NOT ASSUME. threadpoolctl reports the pools the libraries actually created; an
    # env var that arrived too late to be read looks identical to one that worked.
    try:
        import threadpoolctl
        pools = [(d["user_api"], d["num_threads"]) for d in threadpoolctl.threadpool_info()]
        over = [p for p in pools if p[1] > int(_CT)]
        print(f"[shard {a.shard}] thread pools {pools} (cap {_CT})"
              + ("  WARNING: a pool exceeds the cap" if over else ""), flush=True)
    except ImportError:
        print(f"[shard {a.shard}] threadpoolctl absent; cap {_CT} set via env only", flush=True)

    skip = set(x for x in a.skip.split(",") if x)
    want = [r["pdb"] for r in csv.DictReader(open(a.manifest)) if r["pdb"] not in skip]
    pairs_by_pdb, _ = interface_chain_pairs(a.pool, a.roles)

    dq = collections.defaultdict(dict)        # iface -> (seed, sample) -> dockq
    for r in csv.DictReader(open(a.results)):
        if r.get("arm") != "free":
            continue
        try:
            dq[r["iface"]][(int(r["seed"]), int(r["sample"]))] = float(r["dockq"])
        except (TypeError, ValueError):
            continue

    todo = [p for p in want if p in pairs_by_pdb]
    todo = todo[a.shard::a.workers] if a.workers > 1 else todo
    if a.limit:
        todo = todo[:a.limit]

    pl = PAELoss()
    rows, t0, skipped = [], time.time(), collections.Counter()
    for i, pdb in enumerate(todo):
        poses, dockq_by_iface, geom = [], collections.defaultdict(list), None
        for seed in SEEDS:
            got = unit_tensors(a.fold, pdb, seed)
            if got is None:
                continue
            asym, hf, fai, rep, tr, lb = got
            if geom is None:
                xyz, resolved = token_geometry(lb, rep)
                geom = (asym, hf, xyz, resolved)
            for p in range(int(lb["n_pose"])):
                samp = int(lb["sample_index"][p])
                poses.append(pose_bins(pl, tr, lb, p, hf, fai, rep))
                for iface, _ax, _ay in pairs_by_pdb[pdb]:
                    dockq_by_iface[iface].append(dq.get(iface, {}).get((seed, samp), float("nan")))
        if geom is None or not poses:
            skipped["no tensors"] += 1
            continue
        asym, hf, xyz, resolved = geom
        for iface, ax, ay in pairs_by_pdb[pdb]:
            d = np.asarray(dockq_by_iface[iface], dtype=np.float64)
            keep_pose = ~np.isnan(d)
            if keep_pose.sum() < a.min_poses:
                skipped[f"<{a.min_poses} poses"] += 1
                continue
            sel = [poses[j] for j in np.nonzero(keep_pose)[0]]
            row = run_interface(pdb, iface, ax, ay, sel, d[keep_pose], xyz, resolved,
                                asym, hf, a.crop_size, a.crops,
                                not a.d0_from_crop, crop_seed(a.seed, iface))
            if row is None:
                skipped["degenerate"] += 1
                continue
            if "skip_reason" in row:
                skipped[row["skip_reason"].split(" across")[0]] += 1
                continue
            rows.append(row)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [shard {a.shard}] {i+1}/{len(todo)} structures, {len(rows)} interface(s), "
                  f"{el/60:.1f} min, eta {el/(i+1)*(len(todo)-i-1)/60:.1f} min", flush=True)

    out = a.out if a.workers == 1 else a.out.replace(".csv", f".s{a.shard}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # A SHARD CAN LEGITIMATELY PRODUCE NOTHING -- 416 of 846 structures have a single trunk seed
    # and are excluded by --min-poses, so a shard that drew only those has no rows. It must still
    # write its file, or the caller waiting for 16 CSVs waits forever.
    FIELDS = ["pdb", "iface", "n_poses", "n_tokens", "n_pair_tokens", "crop_size", "pin_d0",
              "n_crops", "n_crops_valid", "rho_full", "rho_crop_mean", "rho_crop_std",
              "rho_crop_p10", "rho_crop_p90", "delta", "oracle_dockq", "regret_full",
              "regret_crop_mean", "regret_crop_std", "regret_crop_p90",
              "regret_crop_worse_frac", "regret_delta", "n_crops_degenerate",
              "live_rows_mean", "retention_mean", "retention_p10", "kept_tokens_mean"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[shard {a.shard}] {len(rows)} interface(s) -> {out}   skipped {dict(skipped)}",
          flush=True)


if __name__ == "__main__":
    main()
