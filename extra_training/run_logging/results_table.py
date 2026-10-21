#!/usr/bin/env python
"""Write the full-set run's four metrics to a markdown table, for review without a plot.

READS THROUGH run_logging/metrics_io.py, which plot_full_train.py also uses, so parsing, budget
validation and precedence are one decision rather than two that can drift. They had already
drifted: this file used to take every metric json's DockQ regardless of its budget, while the plot
at least warned -- so a budget-100 rescore could have replaced a budget-30 value here while the
shipped-head column beside it stayed at budget 30.

  run_logging/results_table.py x3k2m4+jhgfxy+2w5to5 --out docs/full_train_results.md
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics_io as MIO

# WHAT EACH SEGMENT OF THE LINEAGE ACTUALLY RAN. Throughput and memory are not comparable across
# these, so an epoch's numbers are only interpretable next to the configuration that produced them.
SEGMENTS = {
    "x3k2m4": "checkpointing ON (blocks_per_ckpt=1); per-epoch fingerprint hashing; "
              "prefetch_issue double-counted the hash",
    "jhgfxy": "checkpointing ON; fresh process so hashing re-paid; prefetch choice-cache added",
    "2w5to5": "checkpointing OFF (blocks_per_ckpt=None); persistent fingerprint cache; "
              "empty_cache at the boundary; exclusive timers + residual + memory peaks",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--root", default="training/runs")
    ap.add_argument("--full-steps", type=int, default=0,
                    help="steps that make an epoch COMPLETE. An epoch is 290 full steps plus a "
                         "remainder, 289 optimiser steps in practice. This was 250 to admit a "
                         "short last shard, and that let a MID-EPOCH read through: epoch 3 was "
                         "published as 2.5880 from a 265-step snapshot when its complete value is "
                         "2.6061. A threshold must exclude partial epochs, not only short ones.")
    ap.add_argument("--out", default="docs/full_train_results.md")
    ap.add_argument("--budget", type=int, default=30,
                    help="the seed budget these DockQ numbers must belong to; a metric json at a "
                         "different budget is REJECTED and named, not quietly plotted")
    ap.add_argument("--expect-poses", type=int, default=1500,
                    help="poses a held-out PAE pass should cover (10 structures x 30 seeds x 5)")
    ap.add_argument("--shipped-val-pae", type=float, default=2.9756)
    ap.add_argument("--shipped-dockq", type=float, default=0.2909)
    ap.add_argument("--shipped-frac", type=float, default=0.5556)
    ap.add_argument("--oracle-dockq", type=float, default=0.5968)
    ap.add_argument("--oracle-frac", type=float, default=1.0000)
    a = ap.parse_args()

    m = MIO.load(a.root, a.runs, a.budget, a.full_steps)
    eps = m.epochs()
    L = []
    L.append("# Full training set: the four metrics, per epoch")
    L.append("")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `run_logging/results_table.py` "
             f"from `{a.runs}`, at **budget {a.budget}**. Regenerated automatically when an "
             f"epoch's fourth metric lands. Do not edit.")
    L.append("")
    L.append("## The lineage is one experiment in three parts")
    L.append("")
    L.append("Epoch numbers are continuous, but the segments do NOT share an implementation, so "
             "**throughput and memory are not comparable across them**. The metrics are.")
    L.append("")
    L.append("| segment | epochs | configuration |")
    L.append("|---|---|---|")
    for seg, desc in SEGMENTS.items():
        owned = sorted(e for e in eps if m.segment.get(e) == seg)
        span = "—" if not owned else (f"{owned[0]}" if len(owned) == 1
                                      else f"{owned[0]}–{owned[-1]}")
        L.append(f"| `{seg}` | {span} | {desc} |")
    L.append("")
    L.append("## Epoch 0 is the shipped head")
    L.append("")
    L.append("| metric | epoch 0 | oracle |")
    L.append("|---|---|---|")
    L.append(f"| held-out PAE | {a.shipped_val_pae:.4f} | — |")
    L.append(f"| val10 mean DockQ | {a.shipped_dockq:.4f} | {a.oracle_dockq:.4f} |")
    L.append(f"| val10 frac-acceptable | {a.shipped_frac:.4f} | {a.oracle_frac:.4f} |")
    L.append("| train PAE | **unmeasured** | — |")
    L.append("")
    L.append(f"Budget {a.budget}, 18 interfaces, 10 validation structures. **Not** the budget-100 "
             "figures (0.2626 / 0.5000, oracle 0.6662 / 1.0000) — never quote one against the "
             "other. The shipped head's train PAE over these 488 structures has never been "
             "measured; it needs a `--loss-only` pass over 4,880 units.")
    L.append("")
    L.append("## Per epoch")
    L.append("")
    L.append("| epoch | seg | complete | train PAE | held-out PAE | vs ep0 | val10 DockQ | vs ep0 "
             "| frac-acc | vs ep0 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for e in eps:
        ok = m.complete(e, a.expect_poses)
        t = f"{m.train_pae[e]:.4f}" if e in m.train_pae else "—"
        if e in m.val_pae:
            v = f"{m.val_pae[e][0]:.4f}"; vd = f"{m.val_pae[e][0] - a.shipped_val_pae:+.4f}"
        else:
            v = vd = "—"
        if e in m.dockq:
            q = f"{m.dockq[e][0]:.4f}"; qd = f"{m.dockq[e][0] - a.shipped_dockq:+.4f}"
            fr = f"{m.dockq[e][1]:.4f}"; frd = f"{m.dockq[e][1] - a.shipped_frac:+.4f}"
        else:
            q = qd = fr = frd = "—"
        L.append(f"| {e} | `{m.segment.get(e, '?')}` | {'yes' if ok else '**NO**'} | {t} | {v} | "
                 f"{vd} | {q} | {qd} | {fr} | {frd} |")
    L.append("")
    L.append("Held-out PAE is a loss, so **lower is better** and a negative delta is an "
             "improvement. DockQ and frac-acceptable are the opposite.")
    L.append("")
    L.append("`complete` means all four metrics are present AND the held-out pass covered its "
             f"expected {a.expect_poses} poses. An epoch marked **NO** is mid-boundary or lost a "
             "metric to a failure — its em-dashes are missing measurements, not zeros.")
    L.append("")
    if m.rejected:
        L.append("### Rejected inputs")
        L.append("")
        for path, why in m.rejected:
            L.append(f"- `{path}` — {why}")
        L.append("")
        L.append(f"A metric json at a budget other than {a.budget} is rejected rather than "
                 "plotted: a different budget is a different pool depth, and taking its number "
                 "while the shipped-head column stays at this budget is the comparison the "
                 "derived baseline exists to prevent.")
        L.append("")
    L.append("## Why held-out PAE is the metric to read")
    L.append("")
    L.append("On the 30-structure arm, training PAE fell 38% from epoch 1 to 142 while held-out "
             "PAE ROSE 51% (2.8290 → 4.2639 against 2.9756 untrained), and the DockQ metrics "
             "showed nothing the entire time because ipTM barely orders poses within an interface "
             "— Spearman 0.20 for the shipped head, 0.10 for the trained one. A falling train-PAE "
             "curve is not evidence a run is working.")
    L.append("")
    L.append("**But held-out PAE is necessary, not sufficient.** It is a loss on the statistic "
             "ipTM derives from, not a measure of ranking. DockQ carries about ±0.05 on 18 "
             "interfaces and frac-acceptable only moves in steps of 1/18 = 0.0556, so single-epoch "
             "moves in those columns are noise — which cuts both ways: epoch 2's fall from 10/18 "
             "to 6/18 acceptable is as uninterpretable alone as a rise would be. A paired "
             "per-interface ranking analysis (`analysis/oneoff/05_val10_rank_correlation.py`) is "
             "the thing that would settle whether selection improved, and it needs several epochs "
             "of checkpoints to be worth running.")
    L.append("")
    rows = [(e, MIO.slowest_rank(m, e)) for e in sorted(m.timing)]
    if rows:
        L.append("## Per-epoch wall clock and memory")
        L.append("")
        L.append("| epoch | seg | slowest rank | wall (min) | accounted | residual | peak alloc "
                 "(GB) | peak reserved (GB) | ckpt |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for e, s in rows:
            if not s:
                continue
            acc = "—" if s["accounted"] is None else f"{s['accounted']:.1f}m"
            res = "—" if s["residual"] is None else f"{s['residual']:.1f}m"
            pa = "—" if s["peak_alloc"] is None else f"{s['peak_alloc']:.2f}"
            pr = "—" if s["peak_reserved"] is None else f"{s['peak_reserved']:.2f}"
            L.append(f"| {e} | `{m.segment.get(e, '?')}` | {s['rank']} | {s['wall']:.1f} | {acc} | "
                     f"{res} | {pa} | {pr} | {s['ckpt'] or '—'} |")
        L.append("")
        L.append("**Wall, accounted and residual all come from the SAME rank** — the one with the "
                 "longest wall clock — so accounted + residual equals wall. Taking each column's "
                 "maximum independently produced a row that did not add up under a heading that "
                 "said 'worst rank'. Memory stays an across-rank maximum on purpose: there the "
                 "question is whether ANY rank came near the card.")
        L.append("")
        L.append("`ckpt` is the live `blocks_per_ckpt`: `1` recomputes every pairformer block's "
                 "forward during backward, `None` stores activations instead. The residual is wall "
                 "clock minus the mutually exclusive buckets; `gpu_forward` and `gpu_backward` are "
                 "CUDA-event spans that overlap `pose_total` and are excluded from the sum.")
        L.append("")
        L.append("### Bucket detail, per rank")
        L.append("")
        for e in sorted(m.timing)[-3:]:
            L.append(f"**epoch {e}** (`{m.segment.get(e, '?')}`)")
            L.append("")
            for rk in sorted(m.timing[e]):
                L.append(f"- rank {rk}: {m.timing[e][rk][1]}")
            L.append("")
    MIO.write_atomic(a.out, "\n".join(L) + "\n")
    n_ok = sum(1 for e in eps if m.complete(e, a.expect_poses))
    print(f"wrote {a.out}  ({len(eps)} epoch(s), {n_ok} complete, {len(m.rejected)} rejected)")


if __name__ == "__main__":
    main()
