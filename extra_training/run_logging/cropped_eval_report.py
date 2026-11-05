#!/usr/bin/env python
"""The cropped cells beside the uncropped pair, with paired error bars computed per comparison.

THE CROPPED NUMBER ON ITS OWN IS NOT A RESULT. Cropping raises every score -- the shipped head's
too -- because a smaller token set shrinks ipTM's d0 and makes the ranking problem easier. The
ground-truth-cropped arm read 0.6305 in-sample cropped against 0.5387 uncropped on the SAME head.
So what matters is the ours-minus-shipped delta under a FIXED crop, and then how that delta
compares to the uncropped one.

TWO BASELINES, REPORTED SEPARATELY. Raised in review, and it is the difference between a real
result and a confounded one:

  PRIMARY     ours mean-selected vs shipped MAX-selected, both on the same cropped inputs. The
              shipped head's native readout is its max-pooled ipTM, and every uncropped number in
              this project compares against shipped read that way. Keeping it makes the cropped
              and uncropped columns answer the same question.
  DIAGNOSTIC  ours mean-selected vs shipped MEAN-selected, both cropped. Strictly the fairer
              like-for-like, and the one that says how much of the primary delta is the selection
              rule rather than the head.

The earlier version of this file printed ONE cropped delta -- ours-mean against shipped-mean --
next to an uncropped delta whose baseline is the shipped head's RECORDED ipTM. Those are two
different shipped objects, so the cropped-minus-uncropped gap moved the crop AND the shipped
readout at once and no part of it could be attributed.

ERROR BARS ARE COMPUTED, NOT ASSUMED. The previous version hardcoded a 0.0044 paired s.e. and a
+0.009 bar carried over from an earlier comparison on a different head and a different crop. Those
do not transfer: a crop changes the per-interface spread, which is what sets the s.e. Every delta
printed here gets a paired s.e. from the per-interface differences of ITS OWN comparison, read out
of the evaluator's own rows_*.csv, and the bar is 1.96 of that s.e.

Run:
  python -I run_logging/cropped_eval_report.py --out training/runs/cropped_eval_y8pyuq \
      --rid y8pyuq --epoch 20 --crop-id de5302ec
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCEPTABLE = 0.23

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="the cropped_eval output directory")
ap.add_argument("--rid", required=True)
ap.add_argument("--epoch", type=int, required=True)
ap.add_argument("--crop-id", default="",
                help="the 8-char crop fingerprint cropped_eval.py printed. REQUIRED unless "
                     "--legacy-unfingerprinted: without it the report globbed for any shipped "
                     "pass and took the most recently modified one, so running the label-free "
                     "variant, then the gt_ost variant, then this report would print gt_ost "
                     "numbers under a label-free heading. Raised in review.")
ap.add_argument("--legacy-unfingerprinted", action="store_true",
                help="read passes written before cropped_eval.py fingerprinted the crop, which "
                     "are bare: insample_shipped.json, insample_ours_ep30.json. Only the "
                     "ground-truth-cropped arm's October passes are of that vintage. Refused "
                     "together with --crop-id, because then there are two answers to which "
                     "files to read.")
ap.add_argument("--variant", default="", help="a label for the heading only; it selects nothing")
a = ap.parse_args()

if bool(a.crop_id) == bool(a.legacy_unfingerprinted):
    sys.exit("STOP: pass exactly one of --crop-id <fp> or --legacy-unfingerprinted. The first "
             "binds every file read to one crop; the second reads the pre-fingerprint passes. "
             "Neither would leave the report free to pick a file by modification time, which is "
             "how a gt_ost number ends up under a label-free heading.")

OUT = os.path.join(ROOT, a.out)
POOLS = (("insample", 738), ("valfull", 92))


def _one(tag, who):
    """EXACTLY ONE file, named for this crop. No globbing by mtime, no fallback to a sibling."""
    if a.legacy_unfingerprinted:
        stem = f"{tag}_ours_ep{a.epoch}" if who == "ours" else f"{tag}_shipped"
    else:
        stem = (f"{tag}_ours_ep{a.epoch}_{a.crop_id}" if who == "ours"
                else f"{tag}_shipped_{a.crop_id}")
    p = os.path.join(OUT, stem + ".json")
    if not os.path.exists(p):
        return None, stem
    return json.load(open(p)), stem


def _rows(tag, who, stem):
    """Per-interface selected DockQ, three ways, from the evaluator's own per-pose rows.

    The selection is argmax over a structure's candidate poses, which is what phase_k_eval's _pick
    does: ours by iptm_mean (mean-selected) or iptm (max-selected), shipped by its own column.
    Reading the rows rather than the summary json is what makes a PAIRED error bar possible -- the
    json carries only the pool mean, and an unpaired s.e. over 92 interfaces is roughly twice the
    paired one on this data, which would make a real effect look like noise.
    """
    pats = [os.path.join(OUT, stem, "rows_b30_*_s*.csv"),
            os.path.join(OUT, stem, "rows_b10_*_s*.csv")]
    files = sorted({f for pat in pats for f in glob.glob(pat)})
    if not files:
        return None
    by = collections.defaultdict(list)
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if not (r.get("dockq") or "").strip():
                    continue
                try:
                    dq = float(r["dockq"])
                    mx = float(r["iptm"])
                    mn = float(r["iptm_mean"]) if (r.get("iptm_mean") or "").strip() else None
                except (ValueError, KeyError):
                    continue
                by[r["iface"]].append((mx, mn, dq, int(r["seed"]), int(r["sample"])))
    out = {}
    for ifc, v in by.items():
        # TIE-BREAK EXACTLY AS phase_k_eval._pick DOES -- (score, -seed, -sample) -- or a tie
        # would resolve differently here and the s.e. would describe a selection nobody ran.
        sel_max = max(v, key=lambda t: (t[0], -t[3], -t[4]))[2]
        vm = [t for t in v if t[1] is not None]
        sel_mean = max(vm, key=lambda t: (t[1], -t[3], -t[4]))[2] if len(vm) == len(v) else None
        out[ifc] = (sel_mean, sel_max)
    return out


def _verify(tag, who, sel, summary):
    """Reconstruction must reproduce the evaluator's OWN saved means. Raised in review.

    Everything this report prints -- every delta, every paired s.e., every better/worse split --
    is built by re-selecting poses here rather than by reading the evaluator's numbers. That makes
    a paired error bar possible, and it also means a mistake in THIS file (a tie broken the other
    way, a column read as the wrong one, an interface dropped for a blank dockq) would produce a
    confident table nobody could distinguish from a real one. So the pool mean is recomputed from
    the same selections and checked against the json the evaluator wrote.

    The json rounds to 4 dp, so the tolerance is one rounding step plus slack, and a mismatch is
    FATAL rather than a warning: a report that cannot reproduce the evaluator's own mean has no
    standing to report anything else.
    """
    TOL = 1.5e-4
    # WHICH RECONSTRUCTION MAPS TO WHICH FIELD DEPENDS ON --select-by. mean_dockq is the metric
    # under the pass's own select_by; mean_dockq_select_max is always the max-selected one.
    by_rule = {"mean": [t[0] for t in sel.values() if t[0] is not None],
               "max": [t[1] for t in sel.values()]}
    want = [("mean_dockq", by_rule.get(summary.get("select_by") or "max")),
            ("mean_dockq_select_max", by_rule["max"])]
    for field, vals in want:
        if field not in summary or not vals:
            continue
        got = st.mean(vals)
        ref = float(summary[field])
        if abs(got - ref) > TOL:
            sys.exit(f"STOP: {tag} {who}: this report reconstructs {field}={got:.6f} from "
                     f"{len(vals)} interface(s), but the evaluator saved {ref:.4f} "
                     f"(off by {got - ref:+.6f}, tolerance {TOL:g}). The per-interface selection "
                     f"in cropped_eval_report.py does not reproduce the pass it is describing, so "
                     f"every delta and error bar below it is unreliable. Do not read the table.")
    # AND THE INTERFACE COUNT, which a mask covering the wrong pool would change silently.
    if summary.get("n_interfaces") and len(sel) != int(summary["n_interfaces"]):
        sys.exit(f"STOP: {tag} {who}: rows carry {len(sel)} interface(s), the summary says "
                 f"{summary['n_interfaces']}.")
    return True


def _paired(ours, theirs):
    """Mean paired difference, its s.e., and the 95% bar, over the interfaces BOTH cover."""
    keys = sorted(set(ours) & set(theirs))
    d = [ours[k] - theirs[k] for k in keys]
    if len(d) < 2:
        return None
    se = st.stdev(d) / math.sqrt(len(d))
    return {"n": len(d), "delta": st.mean(d), "se": se, "bar": 1.96 * se,
            "n_better": sum(1 for x in d if x > 1e-9), "n_worse": sum(1 for x in d if x < -1e-9),
            "n_same": sum(1 for x in d if abs(x) <= 1e-9)}


head = f"\n  {a.rid} epoch {a.epoch}"
head += f", {a.variant}" if a.variant else ""
head += f", crop {a.crop_id}" if a.crop_id else ", pre-fingerprint passes"
print(head)

for tag, n_expect in POOLS:
    co, o_stem = _one(tag, "ours")
    cs, s_stem = _one(tag, "shipped")
    if not (co and cs):
        missing = [s for s, d in ((o_stem, co), (s_stem, cs)) if not d]
        print(f"\n  {tag}: no pass on disk for this crop -- missing {', '.join(missing)}.json")
        continue
    for who, d in (("ours", co), ("shipped", cs)):
        if d["n_interfaces"] != n_expect:
            sys.exit(f"STOP: {tag} {who} scored {d['n_interfaces']} interfaces, expected "
                     f"{n_expect}. The mask does not match the pool.")

    print(f"\n  === {tag}, {n_expect} interfaces, cropped ===")
    ro, rs = _rows(tag, "ours", o_stem), _rows(tag, "shipped", s_stem)
    if ro:
        _verify(tag, "ours", ro, co)
    if rs:
        _verify(tag, "shipped", rs, cs)
    if ro and rs:
        print(f"    rows reproduce the evaluator's saved means for both heads "
              f"({len(ro)} and {len(rs)} interfaces)")
        pri = _paired({k: v[0] for k, v in ro.items() if v[0] is not None},
                      {k: v[1] for k, v in rs.items()})
        dia = _paired({k: v[0] for k, v in ro.items() if v[0] is not None},
                      {k: v[0] for k, v in rs.items() if v[0] is not None})
        for name, r in (("PRIMARY    ours mean-sel vs shipped MAX-sel", pri),
                        ("DIAGNOSTIC ours mean-sel vs shipped mean-sel", dia)):
            if not r:
                print(f"  {name}: not computable")
                continue
            verdict = "CLEARS its bar" if r["delta"] >= r["bar"] else "inside the noise"
            print(f"  {name}")
            print(f"    delta {r['delta']:+.4f}  paired s.e. {r['se']:.4f}  bar +{r['bar']:.4f}  "
                  f"n={r['n']}  -- {verdict}")
            print(f"    {r['n_better']} better, {r['n_worse']} worse, {r['n_same']} identical pick")
    else:
        print("  (per-pose rows missing, so no paired error bar; summary means only)")
    print(f"    pool means: ours mean-sel {co['mean_dockq']:.4f}  ours max-sel "
          f"{co.get('mean_dockq_select_max', float('nan')):.4f}  |  shipped mean-sel "
          f"{cs['mean_dockq']:.4f}  shipped max-sel "
          f"{cs.get('mean_dockq_select_max', float('nan')):.4f}")

    # THE UNCROPPED PAIR, from the run's own metric file. Its baseline_mean_dockq is the shipped
    # head's RECORDED ipTM selected over the same interfaces -- a stored number, not a recomputed
    # one, so it is the closest available analogue of the cropped shipped MAX cell but NOT
    # identical to it. The gap below is therefore reported as indicative and not as an estimate.
    up = os.path.join(ROOT, "training", "runs", a.rid, f"metric_{tag}_ep{a.epoch}.json")
    if os.path.exists(up):
        un = json.load(open(up))
        ud = un["mean_dockq"] - un["baseline_mean_dockq"]
        print(f"  UNCROPPED  ours {un['mean_dockq']:.4f} vs shipped-as-recorded "
              f"{un['baseline_mean_dockq']:.4f} = {ud:+.4f}")
        if ro and rs and pri:
            print(f"  cropped PRIMARY minus uncropped: {pri['delta'] - ud:+.4f}  "
                  f"(indicative -- the two shipped baselines are recomputed vs stored)")
    else:
        print(f"  UNCROPPED  no metric_{tag}_ep{a.epoch}.json for {a.rid}")
