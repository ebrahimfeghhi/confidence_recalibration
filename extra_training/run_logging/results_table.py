#!/usr/bin/env python
"""RESULTS.md -- every epoch of an arm, and the held-out points, as a table.

Regenerated at each epoch boundary by push_epoch.sh so the repo carries the numbers and not
only the pictures. Reads the same helpers the plotters do, so the table and the figures cannot
disagree about what an epoch's value is.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epoch_curve as EC      # noqa: E402
import eval_curve as VC       # noqa: E402


def main(rid):
    chain = EC.run_chain(rid)
    means = EC.epoch_means(rid)
    logs = EC.log_series(rid)
    part = EC.partial_epochs(means)
    args = {}
    for r in chain:
        p = os.path.join(EC.ROOT, "training", "runs", r, "args.json")
        if os.path.exists(p):
            args = json.load(open(p))

    out = [f"# {rid} — live results", ""]
    out.append(f"Chain `{' -> '.join(chain)}`. "
               f"lr {args.get('lr')}, {int(float(args.get('epochs', 0)))} epochs, "
               f"eval every {args.get('train_eval_every')}.")
    out.append("")
    out.append("Regenerated at each epoch boundary. Checkpoints, run logs and trajectories are "
               "not in the repo; these are the numbers and the figures.")
    out.append("")
    out.append("## Training")
    out.append("")
    out.append("| epoch | PAE | rel | rho | spread |")
    out.append("|---|---|---|---|---|")
    for e in sorted(means):
        if e in part:
            continue
        g = lambda k: logs.get(k, {}).get(e)          # noqa: E731
        f = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
        out.append(f"| {e} | {means[e][0]:.4f} | {f(g('rel'))} | {f(g('rho'))} | {f(g('spread'))} |")

    shp = VC.shipped()
    out += ["", "## Held out — 83 structures / 141 interfaces, budget 30", ""]
    out.append("| | mean DockQ | max-pooled | frac acc | rho (mean) | rho (max) |")
    out.append("|---|---|---|---|---|---|")
    if shp:
        out.append(f"| shipped | {shp['mean_dockq']:.4f} | {shp['mean_dockq_select_max']:.4f} "
                   f"| {shp['frac_acc']:.4f} | — | 0.1515 |")
    try:
        rho, _ = VC.rho_series(chain)
    except Exception:
        rho = {}
    for r in chain:
        for f_ in sorted(glob.glob(os.path.join(EC.ROOT, "training", "runs", r,
                                                "metric_valfull83_ep*.json"))):
            m = re.search(r"_ep(\d+)\.json$", f_)
            d = json.load(open(f_))
            ep = int(m.group(1))
            rm = rho.get(ep, {})
            out.append(f"| epoch {ep} | {d['mean_dockq']:.4f} | {d['mean_dockq_select_max']:.4f} "
                       f"| {d['frac_acc']:.4f} "
                       f"| {rm.get('mean', float('nan')):.4f} | {rm.get('max', float('nan')):.4f} |")
    out += ["", "![training](figures/%s_train.png)" % rid]
    if os.path.exists(os.path.join(EC.ROOT, "figures", f"{rid}_validation.png")):
        out.append("![held out](figures/%s_validation.png)" % rid)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "87ja6a"))
