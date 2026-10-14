#!/usr/bin/env python
"""What does a diffusion decode cost relative to a recycle, as a function of complex size?

WHY THIS NUMBER MATTERS. The "redock" idea is: fold once at 10 recycles, perturb the antibody x
antigen block of the pair rep, then run k extra recycles + a decode, N times. Its whole claim is
"cheaper than a free fold, so more docking attempts for the same compute". With

    t = seconds per recycle          d_fixed = diffusion setup (the pair_z conditioning prep)
    d_s = seconds per extra sample   f = featurization + embedding (size-dependent, NOT per recycle)

a free fold costs f + 10t + d_fixed + n*d_s, and one redock costs k*t + d_fixed + n*d_s. Redocks per
free fold is then (f + 10t + d_fixed + n*d_s) / (k*t + d_fixed + n*d_s). At k=1, n=5 that ratio is
~5.5 if the decode is cheap and ~2.1 if the decode is half the fold -- a 2.6x swing in the method's
headline. So this is measured, not assumed. (Featurization was measured separately at ~1.3 s of a
~31 s fold, about 4%, so f is small; this script reports it rather than trusting that.)

HOW. No code changes and no new folding path: `folder.py` already loads the checkpoint once and folds
a queue, and Protenix's `PROTENIX_TIMING=1` already prints a CUDA-SYNCHRONISED split per fold:

    [timing] N_cycle=10 N_sample=5 N_token=701 pairformer=18.32s diffusion=9.14s

The sync matters -- without it the trunk's still-running kernels are charged to "diffusion" (see the
comment at protenix/model/protenix.py:887). "pairformer" covers embedding + all N_cycle recycles;
"diffusion" covers the pair_z cache prep + all N_sample samples.

To separate the per-unit slopes from the fixed costs, each complex is folded at several settings and
the slopes come from differences, not from one absolute number:

    t     = (pairformer(nc=10) - pairformer(nc=1)) / 9
    f     = pairformer(nc=1) - t
    d_s   = (diffusion(ns=5) - diffusion(ns=1)) / 4
    d_fix = diffusion(ns=1) - d_s

`--emit` prints the exact commands; nothing here touches a GPU. `--parse` reads the logs those
commands write and fits the table. Complexes are chosen to span the token range, since every pair
tensor is O(N_token^2) and a single mid-size complex would not predict either end.

    ./cost_model.py --emit  --dataset foldbench --n-complexes 6 --out <dir>
    ./cost_model.py --parse --out <dir>
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/ebrahim/tts_abag")

import compare_arms as C  # noqa: E402
from score_experiment import pool_by_seed  # noqa: E402

PY = "/home/ebrahim/Protenix/.venv/bin/python"
FOLDER = "/home/ebrahim/tts_abag/diffusion_commit_clean_v2/folder.py"
# (N_cycle, N_sample): nc1/nc10 give the recycle slope, ns1/ns5 the sample slope. nc=10,ns=5 is the
# production setting and is included as a CHECK: the fitted model must predict it.
GRID = [(1, 1), (10, 1), (1, 5), (10, 5)]
TIMING = re.compile(
    r"\[timing\] N_cycle=(\d+) N_sample=(\d+) N_token=(\d+) "
    r"pairformer=([\d.]+)s diffusion=([\d.]+)s")


def pick_complexes(dataset, k):
    """in: dataset name, how many complexes.  out: [(pdb, tokens)] spanning the token range.

    Tokens, not atoms: the pair rep is [N_token, N_token, c_z], so token count is what drives cost.
    Picked at even quantiles of the token distribution so both ends are represented.
    """
    ds, _, _ = C.arm_defaults(dataset)
    pool, first = pool_by_seed(50, str(ds.pool_csv))
    pdbs = sorted({i.split("_")[0] for i in pool})
    # gemmi, not hand-parsed CIF columns: an earlier version split on whitespace and picked the
    # wrong fields, reporting "7zau: 2 tokens". A token is a polymer residue (or one atom of a
    # non-polymer), which is what gemmi's residue iteration gives directly.
    import gemmi
    toks = {}
    for pdb in pdbs:
        hits = sorted(glob.glob(f"{ds.free}/{pdb}/*/predictions/*.cif"))
        if not hits:
            continue
        try:
            st = gemmi.read_structure(hits[0])
        except Exception:
            continue
        n = sum(len(ch) for ch in st[0]) if len(st) else 0
        if n:
            toks[pdb] = n
    if not toks:
        raise SystemExit("no token counts -- check the free pool path")
    ordered = sorted(toks.items(), key=lambda kv: kv[1])
    idx = [round(i * (len(ordered) - 1) / max(1, k - 1)) for i in range(k)]
    return [ordered[i] for i in sorted(set(idx))]


def emit(A):
    sel = pick_complexes(A.dataset, A.n_complexes)
    os.makedirs(A.out, exist_ok=True)
    ds, _, _ = C.arm_defaults(A.dataset)
    with open(f"{A.out}/queue.tsv", "w") as fh:
        for pdb, _ in sel:
            fh.write(f"free\t{pdb}\t{pdb}\tcost\t{A.seed}\t-\n")
    json.dump({p: t for p, t in sel}, open(f"{A.out}/tokens.json", "w"), indent=1)

    print(f"complexes ({len(sel)}), spanning the token range:")
    for pdb, t in sel:
        print(f"  {pdb:6} {t:5d} tokens")
    print(f"\nqueue: {A.out}/queue.tsv\n")
    print(f"# {len(GRID)} invocations x {len(sel)} complexes = {len(GRID) * len(sel)} folds.")
    print("# Model reload per invocation is OUTSIDE the [timing] lines, so it does not pollute them.")
    print("# RUN ON AN IDLE GPU: another job on the same card contends for SMs and corrupts exactly")
    print("# the numbers this measures.\n")
    for nc, ns in GRID:
        tag = f"nc{nc}_ns{ns}"
        print(f"CUDA_VISIBLE_DEVICES={A.gpu} LAYERNORM_TYPE=openfold "
              f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PROTENIX_TIMING=1 \\\n"
              f"  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\\n"
              f"  {PY} {FOLDER} --queue {A.out}/queue.tsv --out {A.out}/poses_{tag} \\\n"
              f"    --csv {A.out}/results_{tag}.csv --claim-dir {A.out}/claims_{tag} \\\n"
              f"    --dataset {A.dataset} --ncycle {nc} --nsample {ns} --dtype bf16 \\\n"
              f"    > {A.out}/{tag}.log 2>&1")
    print(f"\nthen: ./cost_model.py --parse --out {A.out}")


def parse(A):
    rows = collections.defaultdict(dict)          # tokens -> (nc, ns) -> (pf, df)
    for log in sorted(glob.glob(f"{A.out}/*.log")):
        for line in open(log):
            m = TIMING.search(line)
            if not m:
                continue
            nc, ns, tok, pf, df = (int(m[1]), int(m[2]), int(m[3]), float(m[4]), float(m[5]))
            rows[tok][(nc, ns)] = (pf, df)
    if not rows:
        raise SystemExit(f"no [timing] lines under {A.out} -- was PROTENIX_TIMING=1 set?")

    print("| tokens | f (embed) | t /recycle | d_fix (setup) | d_s /sample | "
          "decode(5) | fold(10,5) | redocks per fold |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    out = []
    for tok in sorted(rows):
        g = rows[tok]
        need = [(1, 1), (10, 1), (1, 5)]
        if not all(k in g for k in need):
            print(f"| {tok} | incomplete: have {sorted(g)} | | | | | | |")
            continue
        t = (g[(10, 1)][0] - g[(1, 1)][0]) / 9
        f = g[(1, 1)][0] - t
        d_s = (g[(1, 5)][1] - g[(1, 1)][1]) / 4
        d_fix = g[(1, 1)][1] - d_s
        decode5 = d_fix + 5 * d_s
        fold = f + 10 * t + decode5
        redock = (1 * t) + decode5                    # k=1 extra recycle + a 5-sample decode
        print(f"| {tok} | {f:.2f}s | {t:.2f}s | {d_fix:.2f}s | {d_s:.2f}s | "
              f"{decode5:.2f}s | {fold:.2f}s | **{fold / redock:.1f}** |")
        out.append((tok, f, t, d_fix, d_s, fold, redock, g.get((10, 5))))

    print("\n  f = embedding + featurization; t = per recycle; d_fix = pair_z conditioning prep;")
    print("  d_s = per diffusion sample. redocks per fold = fold(10,5) / (1 recycle + 5-sample decode).")
    print("\nCHECK -- fitted vs measured at the production setting (nc=10, ns=5):")
    for tok, f, t, d_fix, d_s, fold, _, meas in out:
        if not meas:
            print(f"  {tok:5d} tokens: not folded at (10,5)")
            continue
        pred_pf, pred_df = f + 10 * t, d_fix + 5 * d_s
        print(f"  {tok:5d} tokens: pairformer {pred_pf:6.2f} vs {meas[0]:6.2f}s "
              f"({100 * (pred_pf - meas[0]) / meas[0]:+.0f}%)   "
              f"diffusion {pred_df:6.2f} vs {meas[1]:6.2f}s "
              f"({100 * (pred_df - meas[1]) / meas[1]:+.0f}%)")
    print("  A large error here means the cost is not linear in recycles/samples and the")
    print("  redocks-per-fold column should not be trusted.")


FOLD = re.compile(r"\] (\S+?_\S+?) \[seed:(\d+)\]: N_asym \d+, N_token (\d+)")
FWD = re.compile(r"\] (\S+?_\S+?) \[seed:(\d+)\] succeeded\. Model forward time: ([\d.]+)s")
SEED_DONE = re.compile(r"\] Seed (\d+) completed in ([\d.]+)s")


def read_arm(logdir):
    """in: a run directory.  out: {(target_tag, seed): (N_token, forward_s, total_s)}

    Parsed from the logs the live runs ALREADY write -- no instrumentation, no GPU. "Model forward
    time" is trunk + diffusion; the "Seed N completed" line that follows it adds featurization, so
    total - forward isolates featurization per fold.
    """
    toks, fwd, tot = {}, {}, {}
    for log in sorted(glob.glob(f"{logdir}/w*.log")):
        last = None
        for line in open(log):
            m = FOLD.search(line)
            if m:
                toks[(m[1], int(m[2]))] = int(m[3])
                continue
            m = FWD.search(line)
            if m:
                last = (m[1], int(m[2]))
                fwd[last] = float(m[3])
                continue
            m = SEED_DONE.search(line)
            if m and last and int(m[1]) == last[1]:
                tot[last] = float(m[2])     # follows its own 'succeeded' line
                last = None
    return {k: (toks[k], fwd[k], tot.get(k)) for k in fwd if k in toks}


def from_logs(A):
    """Per-recycle cost from the nc5 and nc10 arms: SAME folds, ncycle 5 vs 10, so the difference
    over 5 recycles is the per-recycle cost -- paired per fold, free, no GPU."""
    FB = "/home/ebrahim/tts_abag/experimental_data/protenix/foldbench"
    a5 = read_arm(f"{FB}/selftmpl_mask_on")
    a10 = read_arm(f"{FB}/selftmpl_mask_on_nc10")
    both = sorted(set(a5) & set(a10))
    print(f"folds parsed: nc5 {len(a5)}, nc10 {len(a10)}, PAIRED {len(both)}\n")
    if not both:
        raise SystemExit("no folds in common yet")

    import statistics as st
    buckets = collections.defaultdict(list)
    for k in both:
        tok, f5, t5 = a5[k]
        tok10, f10, t10 = a10[k]
        if tok != tok10:
            continue
        t = (f10 - f5) / 5.0                      # per recycle
        feat = (t10 - f10) if t10 else None
        buckets[tok // 200 * 200].append((t, f5, f10, feat, tok))

    print("| tokens | n | fwd nc5 | fwd nc10 | t /recycle | 10t | decode+embed | feat | redocks/fold |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b in sorted(buckets):
        v = buckets[b]
        t = st.median(x[0] for x in v)
        f5 = st.median(x[1] for x in v)
        f10 = st.median(x[2] for x in v)
        feats = [x[3] for x in v if x[3] is not None]
        feat = st.median(feats) if feats else float("nan")
        lump = f10 - 10 * t                       # embedding + decode, not separable from these logs
        redock = t + lump                          # UPPER BOUND on a redock: embedding counted in
        print(f"| {b}-{b + 199} | {len(v)} | {f5:.2f}s | {f10:.2f}s | **{t:.2f}s** | {10 * t:.2f}s | "
              f"{lump:.2f}s | {feat:.2f}s | {(f10 + feat) / redock:.1f} |")
    alln = [x for v in buckets.values() for x in v]
    print(f"\n  n={len(alln)} paired folds, {len(buckets)} size buckets. "
          f"Median per-recycle cost {st.median(x[0] for x in alln):.2f}s.")
    print("  t /recycle = (fwd nc10 - fwd nc5)/5, paired per fold. feat = total - forward.")
    print("  'decode+embed' = fwd nc10 - 10t: this LUMPS the diffusion decode with the input")
    print("  embedding, which these logs cannot separate. So redocks/fold here is a LOWER bound")
    print("  (a redock is charged for the embedding it would not actually repeat).")
    print("  To split it, set PROTENIX_TIMING=1 -- Protenix then prints pairformer vs diffusion.")

    # Where PROTENIX_TIMING is on, prefer it: the split comes from WITHIN one fold, so it is immune
    # to the cross-run GPU contention that biases the paired nc5/nc10 estimate above. Measured
    # 2026-10-05: the pairing predicted 28.0s forward at 400-599 tokens, with 10t = 19.3s of recycles
    # alone, while the in-fold timing says pairformer 17.3s + diffusion 6.7s = 24.0s. 10t exceeding
    # the whole pairformer is impossible, so the paired numbers run hot and only these are quoted.
    tl = collections.defaultdict(list)
    for d in (f"{FB}/selftmpl_mask_on_nc10", f"{FB}/selftmpl_mask_on"):
        for log in sorted(glob.glob(f"{d}/w*.log")):
            for line in open(log):
                m = TIMING.search(line)
                if m:
                    tl[(int(m[1]), int(m[2]), int(m[3]) // 200 * 200)].append((float(m[4]), float(m[5])))
    if not tl:
        print("\n  (no [timing] lines yet -- nothing on the in-fold split)")
        return
    print("\nIN-FOLD SPLIT from PROTENIX_TIMING (trustworthy: one fold, CUDA-synced):")
    print("| N_cycle | N_sample | tokens | n | pairformer | diffusion | decode share of forward |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for (nc, ns, b), v in sorted(tl.items()):
        pf = st.median(x[0] for x in v)
        df = st.median(x[1] for x in v)
        print(f"| {nc} | {ns} | {b}-{b + 199} | {len(v)} | {pf:.2f}s | {df:.2f}s | "
              f"{100 * df / (pf + df):.0f}% |")
    print("  pairformer = embedding + all N_cycle recycles; diffusion = pair_z prep + all N_sample.")
    print("  Splitting embedding from the recycles needs a SECOND N_cycle with timing on; until then")
    print("  the redock cost is best obtained by timing the redock configuration itself.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true", help="pick complexes, write the queue, print commands")
    ap.add_argument("--parse", action="store_true", help="fit the cost model from the logs")
    ap.add_argument("--from-logs", action="store_true",
                    help="per-recycle cost from the live nc5/nc10 arms; no GPU, no instrumentation")
    ap.add_argument("--dataset", default="foldbench")
    ap.add_argument("--n-complexes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=777000)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default="/home/ebrahim/tts_abag/experimental_data/cost_model")
    A = ap.parse_args()
    if A.emit:
        emit(A)
    elif A.parse:
        parse(A)
    elif A.from_logs:
        from_logs(A)
    else:
        ap.error("pass --emit, --parse or --from-logs")
