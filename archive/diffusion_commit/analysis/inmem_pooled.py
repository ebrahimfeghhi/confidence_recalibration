#!/usr/bin/env python
"""In-memory self-templating, scored the way the problem is defined: POOLED across seeds.

WHY POOLED. A "confident decoy" is an interface where the most confident pose IN THE WHOLE POOL is
bad although a good pose exists somewhere. Selection therefore happens across all 25 seeds at once,
not within each seed. Scoring per seed cannot see the effect at all -- it misses a reordering BETWEEN
seeds, which on 9h01_AB turned out to be the entire mechanism (per-seed tables called that interface
flat; pooled selection showed 0.005 -> 0.782).

THE ARMS come from one fold each: decode at recycle 10 (at10, the free baseline and the poses that get
templated), splice them back, then continue. at11/at12/at13 are the same fold at successive recycles,
so nothing between the arms differs except recycle count -- no bf16 basin divergence.

DELTA-ipTM. Templating raised the good seed's confidence far more than the decoys' (+0.194 vs +0.014
on 9h01_AB). If that holds generally, the GAIN in ipTM is a better selector than ipTM itself, so it is
reported per arm: rank seeds by (ipTM after - ipTM before) and check whether the top-ranked seed is
good.

    inmem_pooled.py [--decodes csv] [--final csv] [--thresh 0.23]
"""
import argparse
import collections
import csv
import statistics as st
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit")

import paths            # noqa: E402
from fmt import table   # noqa: E402

ARMS = ("at10", "at11", "at12", "at13")


def load(decodes, final):
    """{iface: {arm: {seed: [(iptm, dockq)]}}}"""
    d = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for path, forced in ((decodes, None), (final, "at13")):
        try:
            fh = open(path)
        except FileNotFoundError:
            continue
        for r in csv.DictReader(fh):
            ip = r["iptm"].strip()
            if not ip:
                continue          # a pose with no confidence cannot take part in selection
            d[r["iface"]][forced or r["arm"]][r["seed"]].append((float(ip), float(r["dockq"])))
    return d


def main():
    ap = argparse.ArgumentParser()
    D = f"{paths.DATA_DIR}/inmem"
    ap.add_argument("--decodes", default=f"{D}/decodes.csv")
    ap.add_argument("--final", default=f"{D}/n13.csv")
    ap.add_argument("--thresh", type=float, default=0.23)
    A = ap.parse_args()
    T = A.thresh

    data = load(A.decodes, A.final)
    rows, dsum = [], []
    for iface in sorted(data):
        arms = data[iface]
        if "at10" not in arms:
            continue
        r = [iface, len(arms["at10"])]
        for arm in ARMS:
            seeds = arms.get(arm)
            if not seeds:
                r += [None, None]
                continue
            allp = [p for v in seeds.values() for p in v]
            shipped = max(allp, key=lambda p: p[0])[1]     # pooled: the single most confident pose
            r += [shipped, max(p[1] for p in allp)]        # and the oracle, for headroom
        rows.append(r)

        # delta-ipTM as a selector: rank SEEDS by how much templating raised their confidence
        base = arms.get("at10", {})
        for arm in ARMS[1:]:
            cur = arms.get(arm)
            if not cur:
                continue
            shared = [s for s in cur if s in base]
            if not shared:
                continue
            dl = sorted(((max(p[0] for p in cur[s]) - max(p[0] for p in base[s])), s)
                        for s in shared)
            top = dl[-1][1]
            good = {s for s in shared if max(p[1] for p in base[s]) >= T}
            # Either bucket can be empty -- an interface where every seed is good, or none is --
            # and st.mean([]) raises rather than returning nothing, killing the whole report.
            dec = [x for x, s in dl if s not in good]
            dsum.append([iface, arm, len(shared), len(good),
                         st.mean([x for x, s in dl if s in good]) if good else None,
                         st.mean(dec) if dec else None,
                         "YES" if top in good else "no",
                         max(p[1] for p in cur[top])])

    hdr = ["interface", "seeds"]
    for a in ARMS:
        hdr += [f"{a} SHIPPED", f"{a} oracle"]
    print("POOLED across seeds -- the pose the model's own ipTM would ship\n")
    print(table(hdr, rows, align="l r" + " r" * (2 * len(ARMS)),
                dec={h: 3 for h in hdr if h not in ("interface", "seeds")}))
    ok = {a: sum(1 for r in rows if (r[2 + 2 * i] or 0) >= T) for i, a in enumerate(ARMS)}
    print("\n  interfaces whose shipped pose is acceptable: "
          + "  ".join(f"{a} {ok[a]}/{len(rows)}" for a in ARMS))

    if dsum:
        print("\n\nDELTA-ipTM AS A SELECTOR -- rank seeds by the confidence GAIN from templating\n")
        print(table(["interface", "arm", "seeds", "good seeds", "mean d good", "mean d decoy",
                     "top-gain seed is good?", "its DockQ"], dsum,
                    align="l l r r r r c r",
                    dec={"mean d good": 3, "mean d decoy": 3, "its DockQ": 3}))
        hit = [d for d in dsum if d[6] == "YES"]
        print(f"\n  top-gain seed is a good seed in {len(hit)}/{len(dsum)} (interface, arm) cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
