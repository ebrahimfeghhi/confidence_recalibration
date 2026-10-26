#!/usr/bin/env python
"""Regenerate the live block of extra_training/EXPERIMENTS.md: the overlay figure and the numbers.

    python -I run_logging/experiments_doc.py

WHY A GENERATED BLOCK AND NOT A HAND-EDITED TABLE. The doc is meant to be the clean record, and a
record whose numbers are retyped drifts from the shards the moment an epoch lands. Everything
between AUTO:BEGIN and AUTO:END is rewritten from each run's own train_eval shards and rank_rho
cache on every call; the prose outside those markers is never touched. If the numbers and the
prose ever disagree, the numbers are right.

ARMS ARE DISCOVERED, NOT LISTED. Arm 3 does not exist yet when this is first run, and a hardcoded
rid would silently keep plotting two arms after the third started. An arm is any run whose
args.json carries the signature below and which has at least one evaluated epoch, so the third
appears in the figure and the table by itself.

SHIPPED IS RECOMPUTED from each run's own stored_iptm rows, never hardcoded -- a hardcoded
baseline has been a bug in this project before.
"""
import glob
import json
import math
import os
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import plot_arms as PA  # noqa: E402

DOC = os.path.join(ROOT, "EXPERIMENTS.md")
FIG = os.path.join(ROOT, "figures", "arms_overlay.png")
FIG_REL = "figures/arms_overlay.png"
BEGIN, END = "<!-- AUTO:BEGIN -->", "<!-- AUTO:END -->"

# (label, predicate on args.json) -- ORDER IS THE EXPERIMENT NUMBER in the doc.
def _arms():
    """[(n, label, rid)] for the arms that exist and have been evaluated at least once.

    A resumed arm is named by its NEWEST member, since plot_arms spans the chain from there. The
    share arm is the chain ending in whichever run has pose_perblock_share and the most epochs.
    """
    found = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "training", "runs", "*", "args.json"))):
        rid = os.path.basename(os.path.dirname(f))
        try:
            a = json.load(open(f))
        except Exception:
            continue
        if not float(a.get("alpha_pae") or 0) or a.get("rank_loss"):
            continue
        if str(a.get("train_eval_pool") or "").find("valfull") < 0:
            continue
        if not a.get("pose_perblock"):
            continue
        s = PA.series(rid, 30, 0.23)
        if not s:
            continue
        if a.get("pose_perblock_share"):
            key, n, lab = "share", 1, "1. share x PAE"
        elif a.get("pose_perblock_copy") and not a.get("l2_sp"):
            key, n, lab = "copy", 2, "2. copy x PAE"
        elif a.get("pose_perblock_copy") and a.get("l2_sp"):
            key, n, lab = "l2sp", 3, f"3. full head + L2-SP {a['l2_sp']:g}"
        else:
            continue
        # NEWEST CHAIN MEMBER WINS: plot_arms spans the chain from the rid it is given, so the
        # one with the most epochs is the one that covers the others.
        if key not in found or len(s) > found[key][2]:
            found[key] = (n, lab, len(s), rid)
    return sorted((n, lab, rid) for n, lab, _, rid in found.values())


def _stats(rid):
    s = PA.series(rid, 30, 0.23)
    r = PA.rho_curve(rid, 30)
    pe, tr, vl = PA.pae_loss(rid)
    T, V = dict(zip(pe, tr)), dict(zip(pe, vl))
    e = sorted(s)
    d = [s[x][0] for x in e]
    last = e[-1]
    out = {
        "eps": f"{e[0]}..{last}",
        "n": len(e),
        "dq": s[last][2], "frac": s[last][3],
        "sh_dq": s[last][4], "sh_frac": s[last][5],
        "paired_mean": st.mean(d),
        "paired_last": s[last][0], "paired_se": s[last][1],
        "rho": r[last][0] if last in r else float("nan"),
        "rho_best": max((r[x][0] for x in r), default=float("nan")),
        "sh_rho": next(iter(r.values()))[2] if r else float("nan"),
        "tr_first": T[min(T)] if T else float("nan"),
        "tr_last": T[max(T)] if T else float("nan"),
        "vl_first": V[min(V)] if V else float("nan"),
        "vl_last": V[max(V)] if V else float("nan"),
        "vl_min": min(V.values()) if V else float("nan"),
        "vl_min_ep": min(V, key=V.get) if V else -1,
    }
    return out


def main():
    arms = _arms()
    if not arms:
        print("STOP: no arms found with valfull eval shards")
        return 1
    # THE FIGURE. plot_arms refuses mismatched interface sets, so a pool error cannot be
    # smuggled into the doc by this script.
    spec = [f"{rid}={lab}" for _, lab, rid in arms]
    cmd = [sys.executable, "-I", os.path.join(HERE, "plot_arms.py"), *spec, "--out", FIG]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        print("STOP: plot_arms failed\n" + (p.stdout or "") + (p.stderr or ""))
        return 1

    rows = ["| # | run | epochs | DockQ | frac | rho | paired ΔDockQ (last) | train PAE | val PAE |",
            "|---|---|---|---|---|---|---|---|---|"]
    detail = []
    for n, lab, rid in arms:
        s = _stats(rid)
        rows.append(
            f"| {n} | `{rid}` | {s['eps']} | {s['dq']:.4f} | {s['frac']:.4f} | {s['rho']:.4f} | "
            f"{s['paired_last']:+.4f} ± {s['paired_se']:.4f} | {s['tr_last']:.4f} | "
            f"{s['vl_last']:.4f} |")
        detail.append(
            f"- **{lab}** (`{rid}`, {s['n']} epochs): paired ΔDockQ mean **{s['paired_mean']:+.4f}**; "
            f"rho {s['rho']:.4f} (best {s['rho_best']:.4f}) vs shipped {s['sh_rho']:.4f}; "
            f"train PAE {s['tr_first']:.4f} → {s['tr_last']:.4f}; "
            f"val PAE {s['vl_first']:.4f} → {s['vl_last']:.4f}, min {s['vl_min']:.4f} at epoch "
            f"{s['vl_min_ep']}.")
    sh = _stats(arms[0][2])
    stamp = subprocess.run(["date", "+%Y-%m-%d %-I:%M %p %Z"], capture_output=True, text=True,
                           env={**os.environ, "TZ": "America/New_York"}).stdout.strip()

    block = "\n".join([
        BEGIN,
        "",
        f"![arms overlay]({FIG_REL})",
        "",
        f"*Regenerated {stamp}. Every number below comes from each run's own train_eval shards "
        f"and rank_rho cache; shipped is recomputed from the same rows via `stored_iptm`.*",
        "",
        f"Shipped on `valfull` (92 interfaces, budget 30): DockQ **{sh['sh_dq']:.4f}**, "
        f"frac **{sh['sh_frac']:.4f}**, rho **{sh['sh_rho']:.4f}**.",
        "",
        *rows,
        "",
        *detail,
        "",
        END,
    ])

    doc = open(DOC).read()
    if BEGIN in doc and END in doc:
        pre, rest = doc.split(BEGIN, 1)
        _, post = rest.split(END, 1)
        new = pre + block + post
    else:
        print("STOP: markers missing from EXPERIMENTS.md; refusing to guess where the block goes")
        return 1
    if new != doc:
        open(DOC, "w").write(new)
        print(f"updated {DOC} ({len(arms)} arm(s): {', '.join(r for _, _, r in arms)})")
    else:
        print(f"no change ({len(arms)} arm(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
