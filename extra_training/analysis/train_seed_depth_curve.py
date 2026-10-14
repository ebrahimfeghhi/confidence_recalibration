#!/usr/bin/env python
"""DockQ versus seed depth, 1 to 10 seeds, on the FT3 training cache fold.

    ./train_seed_depth_curve.py                      # training/fold/results.csv
    ./train_seed_depth_curve.py --thresh 0.23 --csv /tmp/curve.csv

WHAT A ROW IS. A seed depth k. For each interface, a draw is a subset of k of its 10 seeds; the pose
you would ship is the one with the highest PAIR ipTM among that draw's 5k poses, and the row reports
what that pose's DockQ is, averaged over draws with every interface weighted equally.

EXACT, NOT BOOTSTRAPPED. At 10 seeds every depth has at most C(10,5) = 252 distinct draws, so the
expectation over draws is computed in closed form instead of sampled. Order an interface's poses by
ipTM descending; pose j is the selected one exactly when its own seed is in the draw and no
higher-ipTM pose's seed is, so it is selected in C(10 - |seeds above it| - 1, k - 1) of the C(10, k)
draws. Summing DockQ against those weights is the exact mean, and the same weighting over an
indicator gives the exact acceptable fraction. No sampling noise, no seed to fix, nothing to bootstrap
-- which also sidesteps the bug the validation curve had, where each reported fraction came from its
own independent resample instead of one draw classified once.

SELECTION IS BY ipTM, NOT ORACLE. The `oracle` columns are the best DockQ present in the draw, under
the identical weighting, and are a ceiling for reference only: nothing can pick them without the
answer. The gap between the two is the selection failure, and it is the quantity this project exists
to close.

PAIR ipTM, PER INTERFACE. results.csv carries one row per (interface, seed, sample) with that
interface's own pair ipTM, so no global ipTM is involved anywhere.

ONLY COMPLETE INTERFACES. The subset arithmetic above assumes all 10 seeds are present, so an
interface missing any is excluded and counted in the output rather than averaged over a smaller
universe.
"""
import argparse
import collections
import csv
import math
import os

# the repo root, from this file, so renaming the checkout needs no edit here
_REPO = str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path, iface_of, iptm_col, keep=None):
    """out: ({iface: {seed: [(iptm, dockq)]}}, rows read, rows the filter dropped)

    `iface_of` builds the interface key from a row -- our folds carry one `iface` column, the
    benchmark tables carry `pdb, chain1, chain2`. `keep` is the counting rule: a benchmark row has a
    `tier`/`block` and not every block belongs in a baseline.
    """
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    n = dropped = 0
    for r in csv.DictReader(open(path)):
        if keep is not None and not keep(r):
            dropped += 1
            continue
        try:
            ip, dq = float(r[iptm_col]), float(r["dockq"])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(ip) and math.isfinite(dq)):
            continue
        out[iface_of(r)][r["seed"].strip()].append((ip, dq))
        n += 1
    return out, n, dropped


def weights(poses, key, n_seed, k):
    """P(each pose is the one `key` picks) over all C(n_seed, k) draws of k seeds.

    `poses` is [(seed, iptm, dockq)]; `key` orders them best-first. A pose wins a draw exactly when
    its seed is in the draw and no better pose's seed is, so it wins C(n_seed - |better seeds| - 1,
    k - 1) of them. Ties are broken by the order `sorted` leaves, which is stable, so exactly one
    pose wins any draw and the weights sum to 1.

    FIVE POSES SHARE A SEED, which is what the first version got wrong. It treated `above` as the
    seeds to exclude and still gave a pose positive weight when a BETTER POSE OF ITS OWN SEED had
    already been seen -- but then drawing that seed brings the better pose too, so the pose can never
    be selected. Every one of an interface's 50 poses got weight, the weights summed to about 5, and
    the curve reported DockQ 1.78 and 234% acceptable. Only the best pose per seed can ever win, and
    the assertion below is what makes that visible rather than plausible.
    """
    order = sorted(poses, key=key)
    tot = math.comb(n_seed, k)
    above, out = set(), []
    for seed, ip, dq in order:
        if seed in above:
            out.append((0.0, dq))               # a better pose of this same seed always beats it
            continue
        free = n_seed - len(above) - 1
        w = math.comb(free, k - 1) / tot if 0 <= k - 1 <= free else 0.0
        out.append((w, dq))
        above.add(seed)
    tw = sum(w for w, _ in out)
    if abs(tw - 1.0) > 1e-9:
        raise AssertionError(f"draw weights sum to {tw}, not 1, at k={k} over {len(poses)} poses: "
                             "exactly one pose is selected in each draw, so they must")
    return out


DATASETS = {
    # our own folds: one `iface` column, 10 seeds, every row counts
    "train": dict(results=f"{ROOT}/training/fold/results.csv", iptm="iptm",
                  iface=lambda r: r["iface"].strip(), keep=None, seeds=10),
    "val2": dict(results=f"{ROOT}/val2_fold/results.csv", iptm="iptm",
                 iface=lambda r: r["iface"].strip(), keep=None, seeds=100),
    # the benchmarks: pdb + chain pair, and a tier/block that decides what counts
    # THE SINGLE BASELINE TABLE, not the old per-block pool. foldbench_baseline_results.csv already
    # embodies the counting decision -- matched is in, other_seeds is out -- so no `tier` filter
    # applies and `keep` is None. Provenance per row is the `metadata` column.
    "foldbench": dict(results=f"{_REPO}/experimental_data/protenix/foldbench/"
                              "foldbench_baseline_results.csv", iptm="pair_iptm",
                      iface=lambda r: r["iface"],
                      keep=None, seeds=None),
    "pxm2025": dict(results=f"{_REPO}/experimental_data/protenix/pxm2025/"
                            "free_pool/results.csv", iptm="pair_iptm",
                    iface=lambda r: f"{r['pdb']}_{r['chain1']}{r['chain2']}",
                    keep="rule", seeds=None),
}

# FoldBench's `matched` block (seeds 151-200) is tier `suspect`: for some complexes it disagrees with
# the 50-seed pool on how often a seed finds an acceptable pose. The stored exclusion list is these
# COMPLEXES -- build_topup_queue.py:40, which is what actually built the top-up queue, so it is the
# operative definition whatever its provenance.
FLAGGED = ("8up2", "7x8p", "8vyl", "8oni", "7zoz", "8be3", "8urf", "8che", "8byu", "9fww")

COUNTING = {
    # what every published FoldBench number uses: the pool only
    "pool": lambda r: r.get("tier") == "pool",
    # the chosen rule for the long top-off: the whole suspect block stays out
    "strict": lambda r: r.get("tier") != "suspect",
    # the SHORTER top-off: credit `matched` for the complexes that agree with the pool, and keep
    # `other_seeds` out -- its sampling is uneven by construction, a different problem from this one
    "shorter": lambda r: (r.get("tier") != "suspect"
                          or (r.get("block") == "matched" and r.get("pdb") not in FLAGGED)),
    "everything": lambda r: True,
}


def three_classes(poses, n_seed, k, thresh):
    """(acceptable, decoy, no_acceptable) for one interface at depth k, EXACTLY.

    ONE MODEL, THREE FRACTIONS, summing to 1 by construction -- not three independent resamples.
    That bug was in the validation curve: each class drew its own seeds, so a row was three unrelated
    draws and the fractions did not describe one experiment.

      no acceptable : no seed in the draw carries an acceptable pose, which is
                      C(n - n_acc, k) / C(n, k) over the n_acc seeds that do        -- a SAMPLING failure
      acceptable    : the top pair-ipTM pose of the draw clears the threshold       -- the shipped pose is right
      decoy         : whatever is left -- an acceptable pose was drawn and ipTM
                      preferred a worse one                                         -- a SELECTION failure
    """
    acc_seeds = {sd for sd, _ip, dq in poses if dq >= thresh}
    no_acc = (math.comb(n_seed - len(acc_seeds), k) / math.comb(n_seed, k)
              if n_seed - len(acc_seeds) >= k else 0.0)
    ok = sum(w for (w, dq) in weights(poses, lambda t: (-t[1], t[0]), n_seed, k) if dq >= thresh)
    return ok, 1.0 - ok - no_acc, no_acc


def plot(rows, out, thresh, title):
    """The three classes against seed depth, log x -- on a linear axis every curve's movement is
    squeezed into the first centimetre."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, lbl, c in (("acceptable", f"acceptable (shipped pose DockQ >= {thresh})", "#2a7ab0"),
                        ("decoy", "decoy (acceptable pose drawn, ipTM picked worse)", "#d1495b"),
                        ("no_acceptable", "no acceptable pose in the draw", "#8d8d8d")):
        ax.plot(ks, [r[key] for r in rows], marker="o", ms=3, lw=1.8, color=c, label=lbl)
    ax.set_xscale("log")
    ax.set_xlabel("seed depth k (number of seeds drawn, log scale)")
    ax.set_ylabel(f"% of interfaces (n = {rows[0]['n_iface']})")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="center right", framealpha=0.9)
    fig.text(0.01, 0.01, "a dot is one interface, averaged over every draw of k of its seeds; "
                         "interfaces weighted equally", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=170)
    print(f"wrote {out}.png and {out}.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="train", choices=sorted(DATASETS))
    ap.add_argument("--results", default=None, help="overrides the dataset's table")
    ap.add_argument("--counting", default="shorter", choices=sorted(COUNTING),
                    help="benchmark tables only: which blocks count")
    ap.add_argument("--thresh", type=float, default=0.23, help="DockQ for 'acceptable'")
    ap.add_argument("--high", type=float, default=0.80, help="DockQ for 'high quality'")
    ap.add_argument("--seeds", type=int, default=None,
                    help="the depth to report up to; default the dataset's, else the deepest "
                         "universe every included interface has")
    ap.add_argument("--min-seeds", type=int, default=None,
                    help="an interface needs at least this many counted seeds; default --seeds")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--plot", default=None, help="path prefix for .png/.pdf")
    a = ap.parse_args()

    spec = DATASETS[a.dataset]
    path = a.results or spec["results"]
    keep = COUNTING[a.counting] if spec["keep"] == "rule" else None
    data, nrow, dropped = load(path, spec["iface"], spec["iptm"], keep)
    depth = {i: len(sd) for i, sd in data.items()}
    kmax = a.seeds or spec["seeds"] or (min(depth.values()) if depth else 0)
    need = a.min_seeds if a.min_seeds is not None else kmax

    full = {i: s for i, s in data.items() if len(s) >= need}
    npose = sum(len(v) for s in full.values() for v in s.values())
    rule = f", counting rule '{a.counting}'" if keep else ""
    print(f"{path}{rule}")
    print(f"  {nrow} scored rows kept" + (f", {dropped} dropped by the rule" if keep else ""))
    print(f"  interfaces: {len(data)} present, {len(full)} with >= {need} counted seeds")
    if full:
        d = sorted(len(s) for s in full.values())
        print(f"  their seed depth: min {d[0]}, median {d[len(d)//2]}, max {d[-1]}")
    print(f"  poses in the included set: {npose}")
    print(f"  reporting depths 1..{kmax}\n")
    if not full:
        raise SystemExit(f"STOP: no interface has {need} counted seeds")

    by_iface = {i: [(sd, ip, dq) for sd, ps in s.items() for ip, dq in ps]
                for i, s in full.items()}
    nseed = {i: len(s) for i, s in full.items()}

    # EVERY DEPTH, not a chosen few: the table is read for where the curve flattens.
    ks = [k for k in range(1, kmax + 1)]
    rows = []
    for k in ks:
        acc = dec = noa = sel_dq = sel_hi = orc_dq = 0.0
        for i, poses in by_iface.items():
            n = nseed[i]
            x, y, z = three_classes(poses, n, k, a.thresh)
            acc += x; dec += y; noa += z
            for w, dq in weights(poses, lambda t: (-t[1], t[0]), n, k):
                sel_dq += w * dq
                sel_hi += w * (dq >= a.high)
            for w, dq in weights(poses, lambda t: (-t[2], t[0]), n, k):
                orc_dq += w * dq
        n = len(by_iface)
        rows.append({"k": k, "n_iface": n, "acceptable": 100 * acc / n, "decoy": 100 * dec / n,
                     "no_acceptable": 100 * noa / n, "sel_dockq": sel_dq / n,
                     "sel_high": 100 * sel_hi / n, "oracle_dockq": orc_dq / n})

    print(f"x = seed depth k; y = % of interfaces in each class, or the DockQ of the shipped pose.")
    print(f"A dot is one interface (n = {rows[0]['n_iface']}), averaged over every draw of k of its "
          f"seeds, interfaces weighted equally. Selection is top PAIR ipTM; the three classes sum "
          f"to 100 by construction.\n")
    print(f"| seeds | n interfaces | % acceptable | % decoy | % no acceptable | DockQ selected | "
          f"% >= {a.high} selected | DockQ oracle |")
    print("|---|---|---|---|---|---|---|---|")
    show = [k for k in (1, 2, 3, 5, 10, 20, 30, 50, 60, 75, 100) if k <= kmax]
    if kmax not in show:
        show.append(kmax)
    for r in rows:
        if r["k"] not in show:
            continue
        print(f"| {r['k']} | {r['n_iface']} | {r['acceptable']:.1f} | {r['decoy']:.1f} | "
              f"{r['no_acceptable']:.1f} | {r['sel_dockq']:.3f} | {r['sel_high']:.1f} | "
              f"{r['oracle_dockq']:.3f} |")
    for r in rows:
        tot = r["acceptable"] + r["decoy"] + r["no_acceptable"]
        if abs(tot - 100) > 1e-6:
            raise AssertionError(f"the three classes sum to {tot} at k={r['k']}, not 100")
    f, g = rows[0], rows[-1]
    print(f"\n1 -> {kmax} seeds: acceptable {f['acceptable']:.1f}% -> {g['acceptable']:.1f}% "
          f"({g['acceptable']-f['acceptable']:+.1f}), decoy {f['decoy']:.1f}% -> {g['decoy']:.1f}% "
          f"({g['decoy']-f['decoy']:+.1f}), no acceptable {f['no_acceptable']:.1f}% -> "
          f"{g['no_acceptable']:.1f}% ({g['no_acceptable']-f['no_acceptable']:+.1f})")
    print(f"at {kmax} seeds the coverage ceiling is {100-g['no_acceptable']:.1f}% and selection "
          f"delivers {g['acceptable']:.1f}%, so {g['decoy']:.1f} points are lost to selection")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        print(f"wrote {a.csv}")
    if a.plot:
        plot(rows, a.plot, a.thresh,
             f"{a.dataset}: seed depth vs outcome  (n = {rows[0]['n_iface']} interfaces, "
             f"counting rule '{a.counting}')")


if __name__ == "__main__":
    main()
