#!/usr/bin/env python
"""Confident-decoy readout of the seed_scatter_50 refolds, and the experiment's write-up.

SUMMARY
    Reads seed_scatter_score.py's outputs (scatter_summary.csv, matrices/) and each experiment's
    selection.csv, queue.tsv and results.csv. Per dataset:

    1. Per pick, on CONFIDENT decoy interfaces (the original top-ipTM pose has ipTM >= --conf): medians
       over fold x interface rows of agreement, groups, largest-group share, one-offs, share of the 50
       draws acceptable -- plus the committed and scattered population picks for reference.
    2. Routes: per confident decoy interface, over every refolded seed -- is some seed's LARGEST group
       correct (>= --min-group draws, majority acceptable), or is the correct pose only ever a scattered
       draw, or never drawn? For committed-correct cases, decoy ipTM minus the correct group's ipTM.
    3. Paired commitment: on interfaces whose decoy seed still refolds to a WRONG largest group and some
       other refolded seed has a correct group, the decoy seed's largest-group share vs the best correct
       seed's correct-group share.

    Groups = connected components at pose-to-pose DockQ >= 0.5 (as seed_scatter_score.py). Acceptable =
    crystal DockQ >= 0.23. EVALUATION ONLY. Refolds of old pool seeds do not always reproduce the old
    trunk, so a pick's label says how it was CHOSEN, not what its refold did.

    --out writes the full write-up (question, design, status, the three readouts for both datasets,
    findings with their numbers filled in, code and files) as markdown. Rerun it to refresh; the run's
    tmux loop does so every 10 minutes.

INPUTS
    --dataset:   pxm2025 or foldbench, for the stdout readout. Default pxm2025.
    --exp:       the experiment, for the stdout readout. Default as seed_scatter_score.py.
    --conf:      ipTM at which a decoy counts as confident. Default 0.0: every decoy interface (the top-ipTM
                 pose is wrong and an acceptable pose exists), the definition in use since 2026-10-03.
    --min-group: smallest largest-group that counts as committed. Default 10 (of 50).
    --out:       write the write-up for BOTH datasets here instead of printing one readout.

OUTPUTS
    stdout: the readout for --dataset, or with --out the path written

EXAMPLE
    analysis/seed_scatter_report.py
    analysis/seed_scatter_report.py --dataset foldbench
    analysis/seed_scatter_report.py --out experimental_results/seed_scatter_50.md

FUNCTIONS
    groups():   similarity matrix -> component label per pose
    exp_dir():  dataset -> its experiment directory
    status():   one experiment -> folds done, failures, out-of-memory lines
    readout():  one experiment -> the three readouts as data
    render():   one readout -> markdown
    writeup():  both datasets -> the full document
    main():     print one readout, or write the document
"""
import argparse
import collections
import csv
import datetime
import glob
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import paths  # noqa: E402
from fmt import table  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "experimental_data")
PICKS = ("decoy_top_iptm", "decoy_best_dockq", "decoy_committed_correct")
PICK_NAME = {"decoy_top_iptm": "decoy seed (top ipTM)", "decoy_best_dockq": "best-DockQ seed",
             "decoy_committed_correct": "committed-correct seed (old pool)",
             "committed": "population: committed", "scattered": "population: scattered"}


def groups(S):
    """in: similarity matrix.  out: connected-component label per pose at DockQ >= 0.5"""
    lab = -np.ones(len(S), int)
    c = 0
    for i in range(len(S)):
        if lab[i] >= 0:
            continue
        stack, lab[i] = [i], c
        while stack:
            j = stack.pop()
            for m in np.where(S[j] >= 0.5)[0]:
                if lab[m] < 0:
                    lab[m] = c
                    stack.append(m)
        c += 1
    return lab


def exp_dir(dataset):
    """in: a dataset name.  out: its seed_scatter_50 directory"""
    return f"{DATA}/seed_scatter_50" if dataset == "pxm2025" else f"{paths.DATASETS[dataset].data}/seed_scatter_50"


def status(exp):
    """in: an experiment.  out: dict of queue length, folds done, FAILED lines, out-of-memory lines"""
    q = [l.split("\t") for l in open(f"{exp}/queue.tsv") if l.strip()] if os.path.exists(f"{exp}/queue.tsv") else []
    done = {(p, s.strip()) for _, p, _, _, s in q if glob.glob(f"{exp}/poses/{p}/{s.strip()}-*/run.json")}
    logs = "".join(open(f).read() for f in glob.glob(f"{exp}/w[0-9]*.log"))
    ndec = sum(1 for r in csv.DictReader(open(f"{exp}/selection.csv")) if r["pick"].startswith("decoy")) \
        if os.path.exists(f"{exp}/selection.csv") else 0
    dec_folds = list(dict.fromkeys((r["pdb"], r["seed"]) for r in csv.DictReader(open(f"{exp}/selection.csv"))
                                   if r["pick"].startswith("decoy"))) if ndec else []
    return dict(queue=len(q), done=len(done), dec=len(dec_folds), dec_done=sum(f in done for f in dec_folds),
                failed=logs.count("FAILED"), oom=logs.lower().count("out of memory"))


def readout(exp, conf_at=0.5, min_group=10):
    """in: an experiment, the confident-decoy ipTM, the committed group size.
    out: dict with picks rows, routes counts, ipTM gaps, paired commitment rows"""
    sel = list(csv.DictReader(open(f"{exp}/selection.csv")))
    dec = {r["iface"]: (float(r["pose_iptm"]), r["seed"]) for r in sel if r["pick"] == "decoy_top_iptm"}
    conf = {i for i, v in dec.items() if v[0] >= conf_at}
    S = list(csv.DictReader(open(f"{exp}/scatter_summary.csv"))) if os.path.exists(
        f"{exp}/scatter_summary.csv") else []
    cols = ["agreement", "clusters", "top_share", "singletons", "share_acceptable"]
    picks = []
    for k in PICKS + ("committed", "scattered"):
        v = [r for r in S if k in r["pick"].split("+") and (not k.startswith("decoy") or r["iface"] in conf)]
        if v:
            picks.append([PICK_NAME[k], len(v), len({r["pdb"] for r in v})]
                         + [st.median(float(r[c]) for r in v) for c in cols])

    res = {}
    if os.path.exists(f"{exp}/results.csv"):
        for r in csv.DictReader(open(f"{exp}/results.csv")):
            if r["dockq"] and r["iptm"]:
                res[(r["iface"], r["seed"], int(r["sample"]))] = (float(r["iptm"]), float(r["dockq"]))
    per = collections.defaultdict(dict)
    for f in glob.glob(f"{exp}/matrices/*.npz"):
        pdb, seed, iface = os.path.basename(f)[:-4].split("_", 2)
        if iface not in conf:
            continue
        d = np.load(f)
        n = np.where(d["source"] == "new")[0]
        lab = groups(d["dockq"][np.ix_(n, n)])
        v = [res.get((iface, seed, int(k)), (np.nan, 0.0)) for k in d["sample"][n]]
        acc = np.array([x[1] >= 0.23 for x in v])
        ipt = np.array([x[0] for x in v])
        cnt = np.bincount(lab)
        big = cnt.argmax()
        corr = [c for c in range(len(cnt)) if acc[lab == c].mean() > 0.5]
        cbest = max(corr, key=lambda c: cnt[c]) if corr else None
        per[iface][seed] = dict(n=len(n), big=int(cnt[big]), big_correct=acc[lab == big].mean() > 0.5,
                                n_acc=int(acc.sum()), corr=int(cnt[cbest]) if corr else 0,
                                corr_iptm=float(np.nanmedian(ipt[lab == cbest])) if corr else np.nan)
    routes, gaps, pairs = collections.Counter(), [], []
    for i, v in per.items():
        cc = [x for x in v.values() if x["big_correct"] and x["big"] >= min_group]
        if cc:
            routes["committed"] += 1
            gaps.append(dec[i][0] - max(cc, key=lambda x: x["big"])["corr_iptm"])
        elif any(x["n_acc"] for x in v.values()):
            routes["scattered"] += 1
        else:
            routes["none"] += 1
        ds = dec[i][1]
        if ds in v and not v[ds]["big_correct"]:
            others = [x for s, x in v.items() if s != ds and x["corr"] > 0]
            if others:
                b = max(others, key=lambda x: x["corr"])
                pairs.append((i, v[ds]["big"] / v[ds]["n"], b["corr"] / b["n"]))
    return dict(picks=picks, routes=routes, gaps=gaps, pairs=pairs, n_iface=len(per),
                n_cplx=len({i.split("_")[0] for i in per}),
                seeds_med=st.median(len(v) for v in per.values()) if per else 0,
                n_conf=len(conf), n_dec=len(dec), min_group=min_group, conf=conf_at)


def render(R, name):
    """in: a readout, the dataset name.  out: markdown"""
    out = []
    if not R["picks"]:
        return f"No refolds scored yet for {name}.\n"
    which = (f"decoy picks on confident decoys, original ipTM >= {R['conf']}" if R["conf"] > 0 else
             "decoy picks on every decoy interface: top-ipTM pose wrong, an acceptable pose exists")
    out.append(f"**Per pick** ({name}; {which}; "
               "medians over fold x interface rows, 50 draws per fold)\n")
    out.append(table(["seed", "n", "complexes", "agreement", "groups", "largest group share", "one-offs",
                      "share acceptable"], R["picks"],
                     dec={"agreement": 2, "largest group share": 2, "share acceptable": 2, "groups": 0,
                          "one-offs": 0}))
    r = R["routes"]
    out.append(f"\n**How the correct pose appears** ({R['n_iface']} decoy interfaces, "
               f"{R['n_cplx']} complexes, median {R['seeds_med']} refolded seeds each; "
               f"{R['n_conf']} of {R['n_dec']} decoy interfaces pass ipTM >= {R['conf']})\n")
    out.append(table(["route", "interfaces"],
                     [[f"a seed's largest group is correct (>= {R['min_group']} of 50)", r["committed"]],
                      ["correct only as scattered draws", r["scattered"]],
                      ["no correct draw in any refolded seed", r["none"]]], align="l r"))
    if R["gaps"]:
        g = R["gaps"]
        out.append(f"\nWhere a seed commits to the correct pose, the decoy's ipTM is higher than the correct "
                   f"group's median ipTM by {st.median(g):.2f} (median; range {min(g):.2f} to {max(g):.2f}, "
                   f"n={len(g)}).")
    P = R["pairs"]
    if P:
        out.append(f"\n**Paired commitment** ({len(P)} interfaces, {len({p[0].split('_')[0] for p in P})} "
                   "complexes: the decoy seed refolds to a wrong largest group and another refolded seed has "
                   "a correct group)\n")
        out.append(table(["", "median share of the 50 draws"],
                         [["decoy seed, its (wrong) largest group", st.median(p[1] for p in P)],
                          ["best correct seed, its correct group", st.median(p[2] for p in P)]],
                         align="l r", dec={"median share of the 50 draws": 2}))
        out.append(f"\nDecoy seed more committed on {sum(p[1] > p[2] for p in P)}, equal on "
                   f"{sum(p[1] == p[2] for p in P)}, correct seed more committed on {sum(p[1] < p[2] for p in P)}.")
    return "\n".join(out) + "\n"


def writeup(path, conf_at, min_group):
    """in: output path, readout settings.  out: writes the document"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    R = {d: readout(exp_dir(d), conf_at, min_group) for d in ("pxm2025", "foldbench")
         if os.path.exists(f"{exp_dir(d)}/selection.csv")}
    Sx = {d: status(exp_dir(d)) for d in R}
    px = R["pxm2025"]
    pk = {p[0]: p for p in px["picks"]}
    dseed, bseed = pk.get(PICK_NAME["decoy_top_iptm"]), pk.get(PICK_NAME["decoy_best_dockq"])
    com, sca = pk.get(PICK_NAME["committed"]), pk.get(PICK_NAME["scattered"])
    r = px["routes"]
    tot = sum(r.values()) or 1
    P = px["pairs"]

    find = []
    if dseed:
        find.append(f"- **Decoys come from committed seeds.** Refolded with 50 draws, a decoy "
                    f"seed puts a median {dseed[5]:.0%} of its draws in one (wrong) group with "
                    f"{dseed[6]:.0f} one-offs and {dseed[7]:.0%} acceptable (n={dseed[1]}, {dseed[2]} complexes)"
                    + (f"; the committed population sits at {com[5]:.0%}." if com else "."))
    if bseed:
        find.append(f"- **The seed behind the best pose usually does not commit.** Median largest group "
                    f"{bseed[5]:.0%}, {bseed[4]:.0f} groups, {bseed[6]:.0f} one-offs, {bseed[7]:.0%} of draws "
                    f"acceptable (n={bseed[1]})" + (f"; the scattered population: {sca[5]:.0%} largest group, "
                                                    f"{sca[6]:.0f} one-offs." if sca else "."))
    find.append(f"- **Routes to the correct pose** on decoy interfaces: a seed commits to it on "
                f"{r['committed']}/{tot} interfaces ({r['committed'] / tot:.0%}); it appears only as scattered "
                f"draws on {r['scattered']}/{tot} ({r['scattered'] / tot:.0%}); never on {r['none']}/{tot}. "
                f"A lower bound for the committed route: only a median {px['seeds_med']} seeds per interface "
                "are refolded.")
    if px["gaps"]:
        find.append(f"- **Committed-correct is still out-ranked.** The decoy's ipTM beats the correct group's by "
                    f"a median {st.median(px['gaps']):.2f} (n={len(px['gaps'])}).")
    if P:
        find.append(f"- **The decoy seed is the more committed one** on {sum(p[1] > p[2] for p in P)} of "
                    f"{len(P)} paired interfaces (equal {sum(p[1] == p[2] for p in P)}, reverse "
                    f"{sum(p[1] < p[2] for p in P)}); median group share {st.median(p[1] for p in P):.2f} vs "
                    f"{st.median(p[2] for p in P):.2f}. Where both commit fully, ipTM alone prefers the wrong site.")
    if com and sca:
        find.append(f"- **Population: scatter is many separate placements, not one site plus noise.** Committed "
                    f"seeds: {com[5]:.0%} in the largest group; scattered seeds: {sca[4]:.0f} groups, "
                    f"{sca[6]:.0f} one-offs, largest group {sca[5]:.0%} (n={com[1]} / {sca[1]}).")

    stat_rows = []
    for d, s in Sx.items():
        stat_rows.append([d, f"{s['dec_done']}/{s['dec']}", f"{s['done']}/{s['queue']}", s["failed"], s["oom"]])

    doc = f"""# Seed scatter at 50 draws: why confident decoys win

*Auto-generated by `diffusion_commit_clean_v2/analysis/seed_scatter_report.py --out`, refreshed every 10
minutes while the run is going. Last update: {now}. Numbers change as folds land; the findings below are
filled from the current data.*

## Question

When a seed's 5 poses scatter, is the trunk a weak signal (poses land anywhere) or does it encode a few
discrete docking sites? And for **confident decoys** (an acceptable pose exists in the first 50 seeds, but
the top-ipTM pose is wrong): is the decoy seed more committed than the seed that finds the correct pose,
and is the correct pose reached by a committed seed or only by scattered draws?

## Design

- **Refold with 50 draws.** A Protenix trunk is deterministic per seed, so refolding a seed with
  `N_sample=50` draws 50 poses from one trunk and shows the shape of its pose distribution.
- **Seeds** (`selection.csv`), pxm2025 decoys first in the queue:
  - per decoy interface: the seed of its top-ipTM pose (the decoy seed), the seed of its best-DockQ pose,
    and the seed with the most acceptable of its 5 poses in the old pool (committed-correct);
  - population: per antibody-antigen interface, its most committed and most scattered seed by the old
    5-pose agreement;
  - FoldBench: the same three decoy picks (template-free, as FoldBench's pool).
- **Similarity between poses**: DockQ between two predicted poses at one crystal interface (each pose's
  chains from its own crystal mapping), averaged over both directions; OXT atoms left out to match OST.
- **Measures** over a fold's 50 draws: agreement = mean pairwise DockQ; groups = connected components at
  DockQ >= 0.5 (a pose belongs to a group if it matches any member); one-offs = poses with no other pose
  at DockQ >= 0.5; share acceptable = crystal DockQ >= 0.23. Decoy interface = the top-ipTM pose of the first 50
  seeds is wrong and an acceptable pose exists (no ipTM cutoff; `--conf` restricts to confident ones).

## Status

{table(["dataset", "decoy folds done", "all folds done", "FAILED lines", "out-of-memory lines"], stat_rows, align="l r r r r")}

Queue order on GPUs 0-3: pxm2025 decoy picks, then FoldBench decoy picks, then the pxm2025 population.

## Findings so far (pxm2025)

{chr(10).join(find)}

## Results: pxm2025

{render(R['pxm2025'], 'pxm2025')}
## Results: FoldBench

{render(R['foldbench'], 'FoldBench') if 'foldbench' in R else 'Not started.'}
## Caveats

- **Old seeds do not reproduce.** Most of the original 50 seeds per complex were folded by older code with
  unrecorded settings; refolding a seed now can give a different trunk (9g9p seed 300025: 5 scattered wrong
  poses at ipTM ~0.2 then, 50 committed correct poses at ipTM 0.87 now; 8yw9 seed 300022, the old decoy
  seed, now refolds correct). A pick's label says how it was chosen, not what its refold did; readouts use
  the refold itself.
- **Few seeds per interface.** About 3 refolded seeds per decoy interface, so a committed correct seed
  may exist among the others: the committed route is a lower bound.
- **Rows within a complex are not independent**; read n with the complex count.

## Code and files

| file | what it does |
|:---|:---|
| `diffusion_commit_clean_v2/analysis/seed_scatter_select.py` | picks the seeds, writes `selection.csv` and `queue.tsv` (`--dataset`, `--decoys-only`) |
| `diffusion_commit_clean_v2/call_folder.py` / `folder.py` | fold the queue, `--nsample 50`, one tmux worker per GPU, atomic claims |
| `Protenix/protenix/model/modules/confidence.py` | `PROTENIX_CONF_OFFLOAD_TOKENS` (default 2000; this run 0): per-sample pae/pde maps to CPU, so 50 samples fit |
| `diffusion_commit_clean_v2/analysis/seed_scatter_score.py` | 55 x 55 pose-to-pose DockQ per fold and interface (50 new + 5 original poses), `matrices/*.npz`, `scatter_summary.csv` |
| `diffusion_commit_clean_v2/analysis/seed_scatter_report.py` | this document and the stdout readout |
| `diffusion_commit_clean_v2/watch_scatter50.sh` | every 30 min: relaunch dead workers, flag OOM / stalls, score new folds |
| `experimental_data/seed_scatter_50/switch_to_foldbench.sh` | moves GPUs pxm2025 decoys -> FoldBench decoys -> pxm2025 population |

Data: `experimental_data/seed_scatter_50/` (pxm2025) and `experimental_data/protenix/foldbench/seed_scatter_50/`
(FoldBench): `selection.csv`, `queue.tsv`, `poses/`, `results.csv` (crystal DockQ and ipTM per draw),
`matrices/`, `scatter_summary.csv`, `watch.log`.
"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    open(tmp, "w").write(doc)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=paths.DATASET_NAMES, default="pxm2025")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--conf", type=float, default=0.0)
    ap.add_argument("--min-group", type=int, default=10)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    if A.out:
        writeup(A.out, A.conf, A.min_group)
        print(f"  wrote {A.out}")
        return 0
    print(render(readout(A.exp or exp_dir(A.dataset), A.conf, A.min_group), A.dataset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
