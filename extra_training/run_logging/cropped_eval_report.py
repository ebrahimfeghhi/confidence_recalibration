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


def _exact_discordant(b, c):
    """Two-sided exact p on the DISCORDANT pairs only -- McNemar's exact / a paired sign test.

    WHY NOT A NORMAL s.e. FOR frac_acceptable. The indicator differences are 0 for almost every
    interface: on the ground-truth-cropped arm's held-out cell, 4 gained acceptability, 0 lost, 88
    unchanged. A normal approximation on that gives delta +0.0435 against a bar of +0.0419 and
    announces a significant result; the exact test on 4 discordant pairs gives p = 0.125. The
    approximation is not usable at these counts, and this is the same class of error as carrying
    an uncropped error bar onto a cropped comparison -- a bar that does not belong to the quantity
    it is being applied to.

    THE FLOOR THIS IMPLIES, worth knowing before reading any frac_acc number on 92 interfaces:
    with zero losses it takes 6 gains to reach p < 0.05. Five gains is p = 0.0625. So a frac_acc
    move smaller than about +0.065 (6/92) cannot be significant no matter how it is framed.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def _paired(ours, theirs):
    """Paired differences in mean DockQ AND in frac_acceptable, each with its own s.e.

    TWO METRICS, TWO ERROR BARS, and they are not interchangeable. mean DockQ is a continuous
    paired difference, so its s.e. is sd(d)/sqrt(n). frac_acceptable is the fraction of interfaces
    at or above DockQ 0.23, so its paired difference is over per-interface INDICATORS: an interface
    contributes +1 when ours is acceptable and shipped is not, -1 the other way, and 0 when they
    agree -- which is most of them. Using the DockQ spread as the bar for frac_acc, or vice versa,
    would be the same class of error as carrying the uncropped bar onto a cropped comparison.

    The discordant counts are reported too, because a frac_acc delta of +0.0109 means ONE interface
    of 92 changed category, and that is worth seeing rather than inferring from the third decimal.
    """
    keys = sorted(set(ours) & set(theirs))
    if len(keys) < 2:
        return None
    d = [ours[k] - theirs[k] for k in keys]
    se = st.stdev(d) / math.sqrt(len(d))
    # frac_acceptable, paired over indicators
    ai = [(1 if ours[k] >= ACCEPTABLE else 0) - (1 if theirs[k] >= ACCEPTABLE else 0)
          for k in keys]
    a_se = st.stdev(ai) / math.sqrt(len(ai)) if len(set(ai)) > 1 else 0.0
    return {"n": len(d), "delta": st.mean(d), "se": se, "bar": 1.96 * se,
            "n_better": sum(1 for x in d if x > 1e-9), "n_worse": sum(1 for x in d if x < -1e-9),
            "n_same": sum(1 for x in d if abs(x) <= 1e-9),
            "fa_ours": sum(1 for k in keys if ours[k] >= ACCEPTABLE) / len(keys),
            "fa_theirs": sum(1 for k in keys if theirs[k] >= ACCEPTABLE) / len(keys),
            "fa_delta": st.mean(ai), "fa_se": a_se, "fa_bar": 1.96 * a_se,
            "fa_gained": sum(1 for x in ai if x > 0), "fa_lost": sum(1 for x in ai if x < 0),
            "fa_p": _exact_discordant(sum(1 for x in ai if x > 0),
                                      sum(1 for x in ai if x < 0))}


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
            # THE VERDICT IS THE EXACT TEST, not the normal bar. The bar is still printed for
            # reference, and where the two disagree the exact one is the answer.
            fa_verdict = ("significant, exact p=%.4f" % r["fa_p"] if r["fa_p"] < 0.05
                          else "NOT significant, exact p=%.4f" % r["fa_p"])
            print(f"  {name}")
            print(f"    mean DockQ  {r['delta']:+.4f}  paired s.e. {r['se']:.4f}  "
                  f"bar +{r['bar']:.4f}  n={r['n']}  -- {verdict}")
            print(f"      {r['n_better']} better, {r['n_worse']} worse, "
                  f"{r['n_same']} identical pick")
            print(f"    frac_acc    {r['fa_ours']:.4f} vs {r['fa_theirs']:.4f} = "
                  f"{r['fa_delta']:+.4f}  (normal bar +{r['fa_bar']:.4f}, not used)  "
                  f"-- {fa_verdict}")
            print(f"      {r['fa_gained']} interface(s) gained acceptability, "
                  f"{r['fa_lost']} lost it, {r['n'] - r['fa_gained'] - r['fa_lost']} unchanged")
    else:
        print("  (per-pose rows missing, so no paired error bar; summary means only)")
    print(f"    pool means: ours mean-sel {co['mean_dockq']:.4f} (fa {co['frac_acc']:.4f})  "
          f"ours max-sel {co.get('mean_dockq_select_max', float('nan')):.4f} "
          f"(fa {co.get('frac_acc_select_max', float('nan')):.4f})")
    print(f"                shipped mean-sel {cs['mean_dockq']:.4f} (fa {cs['frac_acc']:.4f})  "
          f"shipped max-sel {cs.get('mean_dockq_select_max', float('nan')):.4f} "
          f"(fa {cs.get('frac_acc_select_max', float('nan')):.4f})")

    # THE UNCROPPED PAIR, from the run's own metric file. Its baseline_mean_dockq is the shipped
    # head's RECORDED ipTM selected over the same interfaces -- a stored number, not a recomputed
    # one, so it is the closest available analogue of the cropped shipped MAX cell but NOT
    # identical to it. The gap below is therefore reported as indicative and not as an estimate.
    up = os.path.join(ROOT, "training", "runs", a.rid, f"metric_{tag}_ep{a.epoch}.json")
    if os.path.exists(up):
        un = json.load(open(up))
        ud = un["mean_dockq"] - un["baseline_mean_dockq"]
        print(f"  UNCROPPED  ours {un['mean_dockq']:.4f} vs shipped-as-recorded "
              f"{un['baseline_mean_dockq']:.4f} = {ud:+.4f}   |   frac_acc "
              f"{un['frac_acc']:.4f} vs {un['baseline_frac_acc']:.4f} = "
              f"{un['frac_acc'] - un['baseline_frac_acc']:+.4f}")
        if ro and rs and pri:
            print(f"  cropped PRIMARY minus uncropped: {pri['delta'] - ud:+.4f}  "
                  f"(indicative -- the two shipped baselines are recomputed vs stored)")
    else:
        print(f"  UNCROPPED  no metric_{tag}_ep{a.epoch}.json for {a.rid}")
