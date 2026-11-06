#!/usr/bin/env python
"""THE TWO ACTIVE LOSS TERMS AND THE ipTM-DockQ RELATIONSHIP, per epoch, for an uncropped arm.

WHY THESE TWO AND NOT FIVE. The arm runs --alpha-pae 1.0 and --alpha-rank 75; plddt, pde and
resolved are reported as [DROPPED] in its own banner, so plotting them would imply they are being
optimised. The two that move are the PAE term and the ranking term.

WHERE EACH NUMBER COMES FROM, because they are not the same source and conflating them has bitten
this project before:
  PAE, training       trajectory.csv `pae_loss`, the per-pose value, averaged within an epoch.
  RANKING, training   run.log's per-epoch line, which is the trainer's own accounting over the
                      interface-groups it actually formed pairs for. NOT derivable from
                      trajectory.csv, which carries no ranking column.
  ipTM vs DockQ       the eval shards' rows, `iptm` (the head's prediction) against `dockq`, with
                      `stored_iptm` as the shipped baseline on the SAME poses.

THE CORRELATION IS REPORTED TWO WAYS AND THE WITHIN-INTERFACE ONE IS THE ONE THAT MATTERS.
Pooling every pose of every interface measures mostly which COMPLEX is easy, which the head already
does well -- on d32kqp epoch 66 the across-complex correlation was +0.866 while the median
within-interface Spearman was +0.317. Selection happens inside one interface, so the median
within-interface Spearman is the number tied to what deployment does. Both are drawn.
"""
import argparse, collections, csv, glob, os, re, sys


def spearman(xs, ys):
    """Rank both, then Pearson. Ties take average ranks. No scipy."""
    n = len(xs)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def train_pae(run):
    """{epoch: mean pae_loss} from the trajectory."""
    by = collections.defaultdict(list)
    p = os.path.join(run, "trajectory.csv")
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        for r in csv.DictReader(fh):
            try:
                ep, v = int(r["epoch"]), float(r["pae_loss"])
            except (ValueError, KeyError, TypeError):
                continue
            by[ep].append(v)
    return {e: sum(v) / len(v) for e, v in by.items() if v}


# THE UNIT WORD IS CAPTURED, NOT ASSUMED. The trainer prints "pose(s)" for rel and expected and
# "pair(s)" for the pairwise forms, because rank_loss.rel_coeffs returns a POSE count. An earlier
# version of this file hardcoded "pair(s)", so it silently matched nothing on a rel run and the
# ranking panel came out empty rather than wrong -- which is the better failure, but still wrong.
RANK_RE = re.compile(
    r"\[rank 0\] epoch (\d+) rank-loss \(([^)]*)\): ([+-][\d.]+) over (\d+) interface-group\(s\) "
    r"in (\d+) step\(s\)(?:[^,]*), (\d+) (pose|pair)\(s\)")


def train_rank(run):
    """{epoch: (value, groups, steps, pairs, what)} from the trainer's own per-epoch accounting."""
    out = {}
    p = os.path.join(run, "run.log")
    if not os.path.exists(p):
        return out
    with open(p, errors="replace") as fh:
        for line in fh:
            m = RANK_RE.search(line)
            if m:
                out[int(m.group(1))] = (float(m.group(3)), int(m.group(4)),
                                        int(m.group(5)), int(m.group(6)), m.group(2),
                                        m.group(7) + "(s)")
    return out


def eval_corr(run, pool="valfull"):
    """{epoch: dict} of pooled and median-within-interface correlation, head and shipped."""
    out = {}
    for d in sorted(glob.glob(os.path.join(run, f"{pool}_eval_shards_ep*"))):
        m = re.search(r"_ep(\d+)$", d)
        if not m:
            continue
        ep = int(m.group(1))
        rows = []
        for f in sorted(glob.glob(os.path.join(d, "rows_*.csv"))):
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    try:
                        rows.append((r["iface"], float(r["iptm"]), float(r["dockq"]),
                                     float(r["stored_iptm"])))
                    except (ValueError, KeyError, TypeError):
                        continue
        if len(rows) < 3:
            continue
        pooled_head = spearman([a for _i, a, _d, _s in rows], [d for _i, _a, d, _s in rows])
        pooled_ship = spearman([s for _i, _a, _d, s in rows], [d for _i, _a, d, _s in rows])
        byif = collections.defaultdict(list)
        for i, a, d, s in rows:
            byif[i].append((a, d, s))
        wh, ws = [], []
        for i, v in byif.items():
            if len(v) < 3:
                continue
            c = spearman([x[0] for x in v], [x[1] for x in v])
            k = spearman([x[2] for x in v], [x[1] for x in v])
            if c is not None:
                wh.append(c)
            if k is not None:
                ws.append(k)
        med = lambda L: (sorted(L)[len(L) // 2] if L else None)
        out[ep] = {"pooled_head": pooled_head, "pooled_shipped": pooled_ship,
                   "within_head": med(wh), "within_shipped": med(ws),
                   "n_rows": len(rows), "n_ifaces": len(byif), "n_within": len(wh)}
    return out


def read_form(run):
    """(rank_form, a short descriptor) from the run's own command line, falling back to the log."""
    import json
    cp = os.path.join(run, "config.json")
    cmd = ""
    if os.path.exists(cp):
        try:
            c = json.load(open(cp))
            v = c.get("command") or c.get("passed_through") or ""
            cmd = " ".join(v) if isinstance(v, list) else str(v)
        except Exception:
            cmd = ""
    m = re.search(r"--rank-form\s+(\S+)", cmd)
    form = m.group(1) if m else ""
    if not form:
        lp = os.path.join(run, "run.log")
        if os.path.exists(lp):
            with open(lp, errors="replace") as fh:
                for line in fh:
                    mm = re.search(r"RANKING LOSS on \S+: form (\S+?),", line)
                    if mm:
                        form = mm.group(1)
                        break
    form = form or "unknown"
    bits = []
    for f in ("--alpha-rank", "--rank-within-weight", "--rank-pairs"):
        mm = re.search(re.escape(f) + r"\s+(\S+)", cmd)
        # --rank-pairs and --rank-within-weight are INERT under rel and are not shown for it.
        if mm and not (form == "rel" and f in ("--rank-pairs", "--rank-within-weight")):
            bits.append(f"{f.lstrip('-')} {mm.group(1)}")
    uncropped = not re.search(r"--crop-tokens|--crop-size", cmd)
    bits.append("uncropped" if uncropped else "CROPPED")
    return form, ", ".join(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training/runs/<id>")
    ap.add_argument("--pool", default="valfull")
    ap.add_argument("--out", default="", help="png; default <run>/progress.png")
    a = ap.parse_args()
    run = a.run if os.path.isabs(a.run) else os.path.join("training", "runs",
                                                          os.path.basename(a.run.rstrip("/")))
    pae, rank, corr = train_pae(run), train_rank(run), eval_corr(run, a.pool)
    rid = os.path.basename(run.rstrip("/"))
    # THE OBJECTIVE IS READ FROM THE RUN, NOT WRITTEN INTO THIS FILE. Raised in review: the figure
    # title hardcoded "--rank-pairs all", so a rel run -- which has no pair family -- was labelled
    # with a flag it never used. A progress figure that misnames its own experiment is worse than
    # no figure, because it outlives the conversation that would have corrected it.
    form, cfgnote = read_form(run)

    print(f"run {rid}")
    print(f"{'ep':>3s} {'train PAE':>10s} {'train RANK':>11s} {'groups':>7s} {'size':>9s} "
          f"{'within rho':>11s} {'shipped':>8s} {'pooled rho':>11s}")
    for ep in sorted(set(pae) | set(rank) | set(corr)):
        p = pae.get(ep)
        r = rank.get(ep)
        c = corr.get(ep, {})
        print(f"{ep:3d} "
              f"{(f'{p:.4f}' if p is not None else '-'):>10s} "
              f"{(f'{r[0]:+.5f}' if r else '-'):>11s} "
              f"{(str(r[1]) if r else '-'):>7s} "
              f"{(str(r[3]) if r else '-'):>9s} "
              f"{(f'{c.get(chr(119)+chr(105)+chr(116)+chr(104)+chr(105)+chr(110)+chr(95)+chr(104)+chr(101)+chr(97)+chr(100)):+.4f}' if c.get('within_head') is not None else '-'):>11s} "
              f"{(f'{c[chr(119)+chr(105)+chr(116)+chr(104)+chr(105)+chr(110)+chr(95)+chr(115)+chr(104)+chr(105)+chr(112)+chr(112)+chr(101)+chr(100)]:+.4f}' if c.get('within_shipped') is not None else '-'):>8s} "
              f"{(f'{c[chr(112)+chr(111)+chr(111)+chr(108)+chr(101)+chr(100)+chr(95)+chr(104)+chr(101)+chr(97)+chr(100)]:+.4f}' if c.get('pooled_head') is not None else '-'):>11s}")
    if rank:
        _rk = rank[max(rank)]
        print(f"\nobjective as the trainer reports it: {_rk[4]}")
        print(f"group size unit: {_rk[5]}  (rel and expected count POSES, pairwise forms count "
              f"pairs)")

    if not (pae or rank or corr):
        print("nothing to plot yet")
        return 0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

    if pae:
        e = sorted(pae)
        ax[0].plot(e, [pae[k] for k in e], "o-", color="#1f77b4")
    ax[0].set_title("training PAE loss\n(trajectory.csv, mean per epoch)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("pae_loss"); ax[0].grid(alpha=.3)

    if rank:
        e = sorted(rank)
        ax[1].plot(e, [rank[k][0] for k in e], "s-", color="#d62728")
        ax[1].axhline(0.0, color="k", lw=.8, ls=":")
    _ttl = ("training RELATIVE REGRESSION loss\n(mean-centred vs true iTM; run.log)"
            if form == "rel" else
            f"training RANKING loss ({form})\n(run.log, trainer's own accounting)")
    ax[1].set_title(_ttl)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("ranking loss"); ax[1].grid(alpha=.3)

    if corr:
        e = sorted(corr)
        wh = [corr[k]["within_head"] for k in e]
        ws = [corr[k]["within_shipped"] for k in e]
        ph = [corr[k]["pooled_head"] for k in e]
        ax[2].plot(e, wh, "o-", color="#2ca02c", label="within-interface, head")
        if any(v is not None for v in ws):
            ax[2].axhline([v for v in ws if v is not None][0], color="#7f7f7f", ls="--",
                          label="within-interface, shipped")
        ax[2].plot(e, ph, "^:", color="#9467bd", alpha=.7, label="pooled (across complexes)")
        ax[2].legend(fontsize=8)
    ax[2].set_title("ipTM vs DockQ, Spearman\n(within-interface is what selection uses)")
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("rho"); ax[2].grid(alpha=.3)

    fig.suptitle(f"{rid}: rank-form {form}  |  {cfgnote}  |  mixed depth, 851 structures",
                 fontsize=11)
    fig.tight_layout()
    out = a.out or os.path.join(run, "progress.png")
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
