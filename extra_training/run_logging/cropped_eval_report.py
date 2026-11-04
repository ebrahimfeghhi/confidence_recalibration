#!/usr/bin/env python
"""Print the four cropped cells beside the UNCROPPED pair, because the divergence is the result.

THE CROPPED NUMBER ON ITS OWN IS NOT A RESULT. Cropping raises every score -- the shipped head's
too -- because a smaller token set makes ipTM's d0 smaller and the ranking problem easier. The
ground-truth-cropped arm (o2qczq, resumed as t4jtjr) read 0.6305 in-sample cropped against 0.5387
uncropped on the SAME head. So what this prints is the ours-minus-shipped delta under each crop,
measured against the same 92 held-out interfaces and 738 in-sample ones, plus the gap between
them.

READ IT LIKE THIS. Cropped delta much larger than uncropped is the signature the arm's own
launcher names as the failure to watch: the head has learned to rank CROPS rather than poses, and
only the uncropped column can be deployed. Cropped and uncropped agreeing means the crop was a
convenience, not the mechanism. The held-out paired s.e. is 0.0044, so the bar either column has
to clear is about +0.009.
"""
import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SE, BAR = 0.0044, 0.009

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="the cropped_eval output directory")
ap.add_argument("--rid", required=True)
ap.add_argument("--epoch", type=int, required=True)
ap.add_argument("--variant", default="")
a = ap.parse_args()

OUT = os.path.join(ROOT, a.out)


def _load(*pats):
    """The newest match, so a re-run under a new crop fingerprint does not read a stale pass.

    TWO PATTERNS BECAUSE THERE ARE TWO NAMING SCHEMES. Passes written before cropped_eval.py
    started fingerprinting the crop are bare -- insample_shipped.json, insample_ours_ep30.json --
    and the ground-truth-cropped arm's existing passes are all of that vintage. A single glob with
    a trailing wildcard would have matched ep3 against ep30, so each epoch is spelled out instead.
    """
    hits = []
    for pat in pats:
        hits += glob.glob(os.path.join(OUT, pat))
    hits = sorted(set(hits), key=os.path.getmtime)
    return json.load(open(hits[-1])) if hits else None


print(f"\n  {a.rid} epoch {a.epoch}" + (f", {a.variant} crop" if a.variant else ""))
print(f"  {'':<10} {'CROPPED':>28}   {'UNCROPPED':>28}")
print(f"  {'':<10} {'ours':>8} {'shipped':>8} {'delta':>10}   "
      f"{'ours':>8} {'shipped':>8} {'delta':>10}   {'gap':>8}")

rows = []
for tag, npat in (("insample", 738), ("valfull", 92)):
    co = _load(f"{tag}_ours_ep{a.epoch}_*.json", f"{tag}_ours_ep{a.epoch}.json")
    cs = _load(f"{tag}_shipped_*.json", f"{tag}_shipped.json")
    # THE UNCROPPED PAIR COMES FROM THE RUN'S OWN metric FILE, never from a cropped pass: its
    # baseline_mean_dockq IS the shipped head's uncropped number on the identical interfaces.
    up = os.path.join(ROOT, "training", "runs", a.rid, f"metric_{tag}_ep{a.epoch}.json")
    un = json.load(open(up)) if os.path.exists(up) else None
    if not (co and cs):
        print(f"  {tag:<10} {'-- no cropped pass on disk --':>58}")
        continue
    # A PASS THAT SCORED A DIFFERENT NUMBER OF INTERFACES IS NOT COMPARABLE. A mask that missed
    # its pool would silently shrink or grow this, so it is a stop rather than a footnote.
    for who, d in (("ours", co), ("shipped", cs)):
        if d["n_interfaces"] != npat:
            raise SystemExit(f"STOP: {tag} {who} scored {d['n_interfaces']} interfaces, expected "
                             f"{npat}. The mask does not match the pool.")
    cd = co["mean_dockq"] - cs["mean_dockq"]
    if un:
        ud = un["mean_dockq"] - un["baseline_mean_dockq"]
        print(f"  {tag:<10} {co['mean_dockq']:>8.4f} {cs['mean_dockq']:>8.4f} {cd:>+10.4f}   "
              f"{un['mean_dockq']:>8.4f} {un['baseline_mean_dockq']:>8.4f} {ud:>+10.4f}   "
              f"{cd - ud:>+8.4f}")
        rows.append((tag, cd, ud))
    else:
        print(f"  {tag:<10} {co['mean_dockq']:>8.4f} {cs['mean_dockq']:>8.4f} {cd:>+10.4f}   "
              f"{'-- no uncropped metric file --':>28}")
        rows.append((tag, cd, None))

print(f"\n  held-out paired s.e. {SE:.4f}, so the bar is about +{BAR:.3f}")
for tag, cd, ud in rows:
    if tag != "valfull":
        continue
    verdict = "CLEARS the bar" if cd >= BAR else "inside the noise"
    print(f"  held-out cropped {cd:+.4f} = {cd / SE:.1f} s.e. -- {verdict}")
    if ud is not None:
        print(f"  held-out uncropped {ud:+.4f} = {ud / SE:.1f} s.e.")
        print(f"  the crop is worth {cd - ud:+.4f} of apparent held-out gain on this head")
