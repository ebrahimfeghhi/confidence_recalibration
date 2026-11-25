#!/usr/bin/env python
"""ipTM IS EFFECTIVELY A PER-SEED QUANTITY. The four measurements that establish it.

Everything here is computed from training/fold/results_with_trunks.csv (predicted ipTM and
ground-truth DockQ per pose) and training/fold/itm_labels/<pdb>.npz (ground-truth iTM per pose),
restricted to the 947 interfaces that carry all 15 seeds x 5 samples, so a within-seed and a
between-seed term are both estimable on the same interfaces.

  A  VARIANCE DECOMPOSITION. Inside one interface, split each quantity's variance into a
     between-seed part (the variance of the per-seed means) and a within-seed part (the mean of
     the per-seed variances). The within-seed SHARE is the fraction of what the quantity does
     that a selector choosing among one seed's 5 samples can see.

  B  WITHIN-SEED SELECTION. How often ipTM's argmax among a seed's 5 samples IS that seed's best
     pose by DockQ, against 1/5 = 0.200 for a blind pick, and what DockQ each choice realises.

  C  THE STAKES WITHOUT THE SIGNAL. Among seeds whose 5 poses span a real DockQ range, how often
     ipTM spans almost nothing -- the configuration in which a better within-seed selector would
     pay and the current one cannot.

  D  DEPTH VERSUS BREADTH AT A MATCHED POSE COUNT. 15 poses drawn as 15 seeds x 1 sample, 5 x 3
     and 3 x 5, averaged over repeated draws, scored by ipTM's own pick and by the oracle. Equal
     compute, three different allocations.

WHY D MATTERS MORE THAN A AND B. A and B say the head cannot rank within a seed. D says what that
costs: if the oracle rises with samples per seed while the ipTM pick does not, the extra samples
are generating quality that the confidence head cannot spend, and the same compute spent on more
seeds would have been worth more.

TIE-BREAK, EVERYWHERE: highest score, then lowest seed, then lowest sample -- phase_k_eval.py:795,
so these numbers are comparable with every Success@1 in the repo.
"""
import argparse
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))


def decompose(by_seed):
    """(within_share, between_var, within_var) for {seed: [values]} inside ONE interface.

    total = between + within exactly when the groups are equal-sized, which they are here (5
    samples per seed, interfaces filtered to the complete ones), so the share needs no correction
    term and is not an approximation.
    """
    import numpy as np
    seeds = sorted(by_seed)
    m = np.array([np.mean(by_seed[s]) for s in seeds], dtype=float)
    w = np.array([np.var(by_seed[s]) for s in seeds], dtype=float)
    between, within = float(np.var(m)), float(np.mean(w))
    tot = between + within
    return (within / tot if tot > 0 else float("nan")), between, within


def argmax_like_eval(scores, seeds, samples):
    """Highest score, then LOWEST seed, then LOWEST sample -- phase_k_eval.py:795."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], seeds[i], samples[i]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{ROOT}/training/fold/results_with_trunks.csv")
    ap.add_argument("--itm-dir", default=f"{ROOT}/training/fold/itm_labels")
    ap.add_argument("--seeds", type=int, default=15, help="interfaces must carry exactly this many")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--reps", type=int, default=50, help="random draws per allocation in part D")
    ap.add_argument("--seed-rng", type=int, default=0)
    ap.add_argument("--dockq-spread", type=float, default=0.2, help="part C: a real DockQ range")
    ap.add_argument("--iptm-spread", type=float, default=0.02, help="part C: almost no ipTM range")
    a = ap.parse_args()

    import numpy as np

    # ---------------------------------------------------------------- load
    pose = collections.defaultdict(dict)        # iface -> (seed, sample) -> (dockq, iptm)
    for r in csv.DictReader(open(a.results)):
        if r.get("has_trunk") != "1" or r.get("arm") != "free":
            continue
        try:
            pose[r["iface"]][(int(r["seed"]), int(r["sample"]))] = (
                float(r["dockq"]), float(r["iptm"]))
        except (ValueError, KeyError, TypeError):
            continue

    full = {}
    for iface, d in pose.items():
        seeds = sorted({s for s, _ in d})
        if len(seeds) != int(a.seeds):
            continue
        if any(sum(1 for (s, _) in d if s == sd) != int(a.samples) for sd in seeds):
            continue
        full[iface] = d
    print(f"interfaces with {a.seeds} seeds x {a.samples} samples and a trunk: {len(full)}"
          f"   (of {len(pose)} total)")

    # ground-truth iTM, keyed the same way
    itm = collections.defaultdict(dict)
    for iface in full:
        pdb = iface.split("_")[0]
        p = os.path.join(a.itm_dir, f"{pdb}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        for nm, sd, sp, g in zip(z["iface"], z["seed"], z["sample"], z["g_max"]):
            if str(nm) == iface:
                itm[iface][(int(sd), int(sp))] = float(g)
    # THE iTM STORE HAS ITS OWN SEED SET, AND IT IS SMALLER. training/fold/itm_labels covers
    # seeds 500000-500009 over 370 structures, not the 15 the predicted scores carry, so
    # demanding all 15 matched ZERO interfaces and the row printed nan. The decomposition only
    # needs two or more seeds with a full sample set, so the iTM part uses whatever complete
    # seeds that interface has and reports how many.
    itm_use = {}
    for iface, src in itm.items():
        ok = [sd for sd in sorted({k[0] for k in src})
              if sum(1 for k in src if k[0] == sd) == int(a.samples)]
        if len(ok) >= 2:
            itm_use[iface] = {k: v for k, v in src.items() if k[0] in set(ok)}
    print(f"of those, {len(itm_use)} carry ground-truth iTM on >= 2 complete seeds "
          f"(median {int(np.median([len({k[0] for k in v}) for v in itm_use.values()])) if itm_use else 0} seeds)")

    # ---------------------------------------------------------------- A
    print(f"\n=== A. WITHIN-SEED SHARE OF WITHIN-INTERFACE VARIANCE ===")
    print(f"{'quantity':10s} {'ifaces':>7s} {'mean share':>11s} {'median':>9s}")
    for name in ("dockq", "iptm", "itm_gmax"):
        shares = []
        for iface, d in full.items():
            if name == "itm_gmax":
                src = itm_use.get(iface)
                if not src:
                    continue
                vals = dict(src)
            else:
                j = 0 if name == "dockq" else 1
                vals = {k: v[j] for k, v in d.items()}
            bs = collections.defaultdict(list)
            for (sd, _sp), v in vals.items():
                bs[sd].append(v)
            sh, _b, _w = decompose(bs)
            if sh == sh:
                shares.append(sh)
        print(f"{name:10s} {len(shares):7d} {np.mean(shares):11.4f} {np.median(shares):9.4f}")

    # ---------------------------------------------------------------- B
    print(f"\n=== B. ipTM PICKING THE BEST OF A SEED'S {a.samples} SAMPLES ===")
    hit, n, dq_pick, dq_mean, dq_max = 0, 0, [], [], []
    for iface, d in full.items():
        for sd in sorted({s for s, _ in d}):
            ks = sorted([k for k in d if k[0] == sd], key=lambda k: k[1])
            dqs = [d[k][0] for k in ks]
            ips = [d[k][1] for k in ks]
            i = argmax_like_eval(ips, [k[0] for k in ks], [k[1] for k in ks])
            n += 1
            if dqs[i] >= max(dqs) - 1e-12:
                hit += 1
            dq_pick.append(dqs[i])
            dq_mean.append(float(np.mean(dqs)))
            dq_max.append(max(dqs))
    print(f"seeds: {n}")
    print(f"  ipTM picks the seed's best pose   {hit / n:.4f}   (blind pick {1 / int(a.samples):.4f})")
    print(f"  mean DockQ of ipTM's pick         {np.mean(dq_pick):.4f}")
    print(f"  mean DockQ of a blind pick        {np.mean(dq_mean):.4f}   "
          f"(ipTM - blind = {np.mean(dq_pick) - np.mean(dq_mean):+.4f})")
    print(f"  mean DockQ of the seed's best     {np.mean(dq_max):.4f}   "
          f"(headroom {np.mean(dq_max) - np.mean(dq_pick):+.4f})")
    eff = ((np.mean(dq_pick) - np.mean(dq_mean)) / (np.mean(dq_max) - np.mean(dq_mean))
           if np.mean(dq_max) > np.mean(dq_mean) else float("nan"))
    print(f"  fraction of the within-seed headroom ipTM converts: {eff:.4f}")

    # ---------------------------------------------------------------- C
    print(f"\n=== C. SEEDS WHERE THE STAKES ARE REAL AND ipTM IS FLAT ===")
    tot, wide, wide_flat = 0, 0, 0
    for iface, d in full.items():
        for sd in sorted({s for s, _ in d}):
            ks = [k for k in d if k[0] == sd]
            dqs = [d[k][0] for k in ks]
            ips = [d[k][1] for k in ks]
            tot += 1
            if max(dqs) - min(dqs) > float(a.dockq_spread):
                wide += 1
                if max(ips) - min(ips) < float(a.iptm_spread):
                    wide_flat += 1
    print(f"  seeds whose DockQ spans > {a.dockq_spread}: {wide}/{tot} = {wide / tot:.4f}")
    print(f"  of those, ipTM spans < {a.iptm_spread}:      {wide_flat}/{wide} = "
          f"{wide_flat / wide:.4f}" if wide else "  (none)")

    # ---------------------------------------------------------------- D
    print(f"\n=== D. 15 POSES, THREE ALLOCATIONS ({a.reps} draws each) ===")
    print(f"{'allocation':16s} {'ipTM pick':>10s} {'oracle':>9s} {'blind':>9s} {'top1_hit':>9s}")
    rng = np.random.default_rng(int(a.seed_rng))
    allocs = [(15, 1), (5, 3), (3, 5)]
    for nseed, nsamp in allocs:
        picks, orcs, blinds, hits = [], [], [], []
        for _ in range(int(a.reps)):
            pk, oc, bl, ht = [], [], [], []
            for iface, d in full.items():
                seeds = sorted({s for s, _ in d})
                use_s = rng.choice(seeds, size=nseed, replace=False)
                sc, dq, sds, sps = [], [], [], []
                for sd in use_s:
                    ks = sorted([k for k in d if k[0] == sd], key=lambda k: k[1])
                    take = rng.choice(len(ks), size=nsamp, replace=False)
                    for t in take:
                        k = ks[t]
                        dq.append(d[k][0])
                        sc.append(d[k][1])
                        sds.append(k[0])
                        sps.append(k[1])
                i = argmax_like_eval(sc, sds, sps)
                pk.append(dq[i])
                oc.append(max(dq))
                bl.append(float(np.mean(dq)))
                ht.append(1.0 if dq[i] >= max(dq) - 1e-12 else 0.0)
            picks.append(np.mean(pk))
            orcs.append(np.mean(oc))
            blinds.append(np.mean(bl))
            hits.append(np.mean(ht))
        print(f"{f'{nseed} seeds x {nsamp}':16s} {np.mean(picks):10.4f} {np.mean(orcs):9.4f} "
              f"{np.mean(blinds):9.4f} {np.mean(hits):9.4f}")
    # AND THE SAME SEEDS, 1 SAMPLE VS 5: what the extra samples add at FIXED seed count.
    for nsamp in (1, int(a.samples)):
        picks, orcs = [], []
        for _ in range(int(a.reps)):
            pk, oc = [], []
            for iface, d in full.items():
                sc, dq, sds, sps = [], [], [], []
                for sd in sorted({s for s, _ in d}):
                    ks = sorted([k for k in d if k[0] == sd], key=lambda k: k[1])
                    take = rng.choice(len(ks), size=nsamp, replace=False)
                    for t in take:
                        k = ks[t]
                        dq.append(d[k][0])
                        sc.append(d[k][1])
                        sds.append(k[0])
                        sps.append(k[1])
                i = argmax_like_eval(sc, sds, sps)
                pk.append(dq[i])
                oc.append(max(dq))
            picks.append(np.mean(pk))
            orcs.append(np.mean(oc))
        print(f"{f'{a.seeds} seeds x {nsamp}':16s} {np.mean(picks):10.4f} {np.mean(orcs):9.4f}")


if __name__ == "__main__":
    main()
