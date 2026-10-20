#!/usr/bin/env python
"""Write the full-set run's four metrics to a markdown table, for review without a plot.

ONE SOURCE, TWO RENDERINGS. plot_full_train.py draws the same numbers; this writes them so an
agent or a reviewer can read exact values rather than estimate them off axes. Both read the run
directories, so neither can drift from the other -- a number appearing here that is not in the
plot would mean one of them has a reader bug, not that the run disagrees with itself.

THE LINEAGE IS JOINED. The full-set experiment is x3k2m4 + jhgfxy + 2w5to5, restarted twice for
code fixes, and its epochs number continuously across them. Anything that treats them as three
runs reads three short curves instead of one.

  run_logging/results_table.py x3k2m4+jhgfxy+2w5to5 --out docs/full_train_results.md
"""
import argparse
import csv
import collections
import glob
import json
import os
import re
import time

DOCKQ = re.compile(r"epoch (\d+) TRAIN: mean_dockq ([0-9.]+)\s+frac_acc ([0-9.]+)")
TOOK = re.compile(r"\[rank (\d+)\] epoch (\d+) took ([0-9.]+) min -- (.*)")
RESID = re.compile(r"\[rank (\d+)\] epoch (\d+) accounted ([0-9.]+)m of ([0-9.]+)m, residual ([0-9.-]+)m")
MEMLN = re.compile(r"\[rank (\d+)\] epoch (\d+) memory: peak allocated ([0-9.]+) GB, "
                   r"peak reserved ([0-9.]+) GB, blocks_per_ckpt=(\S+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--root", default="training/runs")
    ap.add_argument("--full-steps", type=int, default=250)
    ap.add_argument("--out", default="docs/full_train_results.md")
    # BUDGET 30 REFERENCES. Not the budget-100 ones (0.2626 / 0.5000, oracle 0.6662 / 1.0000);
    # quoting one against the other is a real error this project has made.
    ap.add_argument("--shipped-val-pae", type=float, default=2.9756)
    ap.add_argument("--shipped-dockq", type=float, default=0.2909)
    ap.add_argument("--shipped-frac", type=float, default=0.5556)
    ap.add_argument("--oracle-dockq", type=float, default=0.5968)
    ap.add_argument("--oracle-frac", type=float, default=1.0000)
    a = ap.parse_args()

    parts = a.runs.split("+")
    tp = collections.defaultdict(lambda: [0.0, 0, 0])
    vp, dq, took, resid, mem = {}, {}, {}, {}, {}
    for part in parts:
        d = os.path.join(a.root, part)
        f = os.path.join(d, "trajectory.csv")
        if os.path.exists(f):
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    if r.get("step_taken") != "1":
                        continue
                    try:
                        e, v, n = int(r["epoch"]), float(r["batch_pae_loss"]), int(r["batch_n"])
                    except (ValueError, KeyError, TypeError):
                        continue
                    tp[e][0] += v * n; tp[e][1] += n; tp[e][2] += 1
        p = os.path.join(d, "val_loss.csv")
        if os.path.exists(p):
            with open(p) as fh:
                for r in csv.DictReader(fh):
                    try:
                        vp[int(r["epoch"])] = (float(r["val_pae"]), int(r["n_poses"]))
                    except (ValueError, KeyError):
                        continue
        log = os.path.join(d, "run.log")
        if os.path.exists(log):
            txt = open(log, errors="replace").read()
            for m in DOCKQ.finditer(txt):
                dq[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
            for m in TOOK.finditer(txt):
                took.setdefault(int(m.group(2)), {})[int(m.group(1))] = (float(m.group(3)), m.group(4))
            for m in RESID.finditer(txt):
                resid.setdefault(int(m.group(2)), {})[int(m.group(1))] = (float(m.group(3)), float(m.group(4)), float(m.group(5)))
            for m in MEMLN.finditer(txt):
                mem.setdefault(int(m.group(2)), {})[int(m.group(1))] = (float(m.group(3)), float(m.group(4)), m.group(5))
        for jp in sorted(glob.glob(os.path.join(d, "metric_train_ep*.json"))):
            try:
                j = json.load(open(jp))
                ep = int(re.search(r"ep(\d+)\.json$", jp).group(1))
            except (ValueError, OSError, AttributeError):
                continue
            dq[ep] = (float(j["mean_dockq"]), float(j["frac_acc"]))

    epochs = sorted(set([e for e, v in tp.items() if v[2] >= a.full_steps]) | set(vp) | set(dq))
    L = []
    L.append("# Full training set: the four metrics, per epoch")
    L.append("")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `run_logging/results_table.py`, "
             f"from `{a.runs}`. Regenerated automatically at each epoch boundary; do not edit.")
    L.append("")
    L.append("**The lineage is one experiment in three parts.** `x3k2m4` ran epoch 1 and died on a "
             "NameError in the held-out monitor. `jhgfxy` resumed from its bundle and ran epochs "
             "2. `2w5to5` resumed from that and runs epoch 3 onward **without activation "
             "checkpointing**. Epoch numbers are continuous.")
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
    L.append("Budget 30, 18 interfaces, 10 validation structures. **Not** the budget-100 figures "
             "(0.2626 / 0.5000, oracle 0.6662 / 1.0000) — never quote one against the other. "
             "The shipped head's train PAE over these 488 structures has never been measured; it "
             "needs a `--loss-only` pass over 4,880 units.")
    L.append("")
    L.append("## Per epoch")
    L.append("")
    L.append("| epoch | train PAE | held-out PAE | vs ep0 | val10 DockQ | vs ep0 | frac-acc | vs ep0 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for e in epochs:
        t = f"{tp[e][0]/tp[e][1]:.4f}" if (e in tp and tp[e][2] >= a.full_steps) else "—"
        if e in vp:
            v = f"{vp[e][0]:.4f}"
            vd = f"{vp[e][0] - a.shipped_val_pae:+.4f}"
        else:
            v = vd = "—"
        if e in dq:
            q = f"{dq[e][0]:.4f}"; qd = f"{dq[e][0] - a.shipped_dockq:+.4f}"
            fr = f"{dq[e][1]:.4f}"; frd = f"{dq[e][1] - a.shipped_frac:+.4f}"
        else:
            q = qd = fr = frd = "—"
        L.append(f"| {e} | {t} | {v} | {vd} | {q} | {qd} | {fr} | {frd} |")
    L.append("")
    L.append("Held-out PAE is a loss, so **lower is better** and a negative delta is an "
             "improvement. DockQ and frac-acceptable are the opposite.")
    L.append("")
    L.append("## Why held-out PAE is the panel to read")
    L.append("")
    L.append("On the 30-structure arm, training PAE fell 38% from epoch 1 to 142 while held-out "
             "PAE ROSE 51% (2.8290 -> 4.2639 against 2.9756 untrained), and the DockQ metrics "
             "showed nothing the entire time because ipTM barely orders poses within an interface "
             "— Spearman 0.20 for the shipped head, 0.10 for the trained one. A falling train-PAE "
             "curve is not evidence a run is working. DockQ also carries about +/-0.05 on 18 "
             "interfaces and frac-acceptable only moves in steps of 1/18 = 0.0556, so single-epoch "
             "moves in those two columns are noise.")
    L.append("")
    if took:
        L.append("## Per-epoch wall clock and memory")
        L.append("")
        L.append("| epoch | wall (min) | accounted | residual | peak alloc (GB) | peak reserved (GB) | ckpt |")
        L.append("|---|---|---|---|---|---|---|")
        for e in sorted(took):
            w = max(v[0] for v in took[e].values())
            if e in resid:
                acc = max(v[0] for v in resid[e].values()); res = max(v[2] for v in resid[e].values())
                accs, ress = f"{acc:.1f}m", f"{res:.1f}m"
            else:
                accs = ress = "—"
            if e in mem:
                pa = max(v[0] for v in mem[e].values()); pr = max(v[1] for v in mem[e].values())
                ck = sorted({v[2] for v in mem[e].values()})[0]
                pas, prs = f"{pa:.2f}", f"{pr:.2f}"
            else:
                pas = prs = ck = "—"
            L.append(f"| {e} | {w:.1f} | {accs} | {ress} | {pas} | {prs} | {ck} |")
        L.append("")
        L.append("Worst rank per epoch. `ckpt` is the live `blocks_per_ckpt`: `1` recomputes every "
                 "pairformer block's forward during backward, `None` stores activations instead. "
                 "The residual is wall clock minus the mutually exclusive buckets; `gpu_forward` "
                 "and `gpu_backward` are CUDA-event spans that overlap `pose_total` and are "
                 "excluded from it.")
        L.append("")
        L.append("### Bucket detail, per rank")
        L.append("")
        for e in sorted(took)[-3:]:
            L.append(f"**epoch {e}**")
            L.append("")
            for rk in sorted(took[e]):
                L.append(f"- rank {rk}: {took[e][rk][1]}")
            L.append("")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {a.out}  ({len(epochs)} epoch(s))")


if __name__ == "__main__":
    main()
