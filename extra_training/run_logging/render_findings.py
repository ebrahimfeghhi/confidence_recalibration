#!/usr/bin/env python3
"""Regenerate an arm's findings doc from its own records. No remembered constants.

    python -I run_logging/render_findings.py --rid nzqhdf [--budget 30] [--out docs/x.md]

Every number here is read back from the run's own files -- the rank_rho cache, run.log, the newest
resume_ep*.pt and the eval shards -- so the doc can be regenerated at any epoch and will agree with
whatever the run has actually written. SHIPPED baselines are derived from the SAME shards via the
`stored_iptm` column (the shipped head's own ipTM for each pose), never typed in from memory: a
baseline carried across runs by hand is how a comparison quietly stops being against the thing it
claims to be against.
"""
import argparse, csv, glob, json, math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_logging import rank_rho as RHO
from run_logging import plot_three as P3
from phases import rank_loss as RKL


def _sel(rows, key):
    """argmax `key` within each interface -> that pose's DockQ."""
    best = {}
    for r in rows:
        k, v = r["iface"], float(r[key])
        if k not in best or v > best[k][0]:
            best[k] = (v, float(r["dockq"]))
    return {k: v[1] for k, v in best.items()}


def shards(rundir, ep):
    rows = []
    for f in glob.glob(f"{rundir}/train_eval_shards_ep{ep}/rows_b30_*.csv"):
        rows += list(csv.DictReader(open(f)))
    return rows


def mean_sem(xs):
    n = len(xs)
    if not n:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, sd / math.sqrt(n)


def paired(rundir, ep_a, ep_b):
    a, b = _sel(shards(rundir, ep_a), "iptm"), _sel(shards(rundir, ep_b), "iptm")
    ks = sorted(set(a) & set(b))
    d = [b[k] - a[k] for k in ks]
    m, se = mean_sem(d)
    movers = sorted(((k, a[k], b[k], b[k] - a[k]) for k in ks), key=lambda t: -abs(t[3]))
    return {
        "n": len(ks), "mean": m, "sem": se,
        "sigma": (abs(m) / se if se else 0.0),
        "up": sum(1 for x in d if x > 1e-9),
        "down": sum(1 for x in d if x < -1e-9),
        "same": sum(1 for x in d if abs(x) <= 1e-9),
        "movers": [t for t in movers if abs(t[3]) > 1e-9][:5],
    }


def drift(rundir):
    pat = re.compile(r"rank 0\] epoch (\d+) l2-sp: \|\|W-W0\|\|\^2 = ([0-9.]+), penalty = ([0-9.]+)")
    out = {}
    with open(f"{rundir}/run.log", errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out


def valloss(rundir):
    pat = re.compile(r"^epoch (\d+) VALLOSS: pae ([0-9.]+)")
    out = {}
    with open(f"{rundir}/run.log", errors="replace") as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def excluded_norms(rundir):
    """Norms of the tensors --l2-sp-exclude kept OUT of the anchor. They are zero-init, so these
    are the direct test of whether the loss finds any use for them."""
    import torch
    cks = glob.glob(f"{rundir}/resume_ep*.pt")
    if not cks:
        return None, {}
    f = max(cks, key=lambda p: int(re.search(r"ep(\d+)", p).group(1)))
    sd = torch.load(f, map_location="cpu", weights_only=True)
    inner = sd.get("head") or sd.get("model") or sd
    args = json.load(open(f"{rundir}/args.json"))
    # The launcher's --note is recorded in config.json, not args.json.
    cfg = {}
    try:
        cfg = json.load(open(f"{rundir}/config.json"))
    except Exception:
        pass
    pats = [p for p in (args.get("l2_sp_exclude") or "").split(",") if p]
    if not pats:
        return os.path.basename(f), {}
    return os.path.basename(f), {
        k: float(v.float().norm()) for k, v in inner.items()
        if any(re.search(p, k) for p in pats)
    }


def trainable_line(rundir):
    """The trainable set AS THE TRAINER REPORTED IT. This is ground truth and the --note is not:
    a launcher's note is prose written before the run, and nzqhdf's still claimed "ALL 248 head
    tensors ... 19.8M parameters" after --grad-probe had refused that set and the arm was narrowed
    to the 100 tensors that actually receive ranking gradient. Read the log, not the intent."""
    pat = re.compile(r"trainable: (\d+) tensor\(s\), ([0-9.]+)M of ([0-9.]+)M parameters \(([0-9.]+)%\)")
    with open(f"{rundir}/run.log", errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                return {"tensors": int(m.group(1)), "millions": float(m.group(2)),
                        "of": float(m.group(3)), "pct": float(m.group(4))}
    return None


def anchored_line(rundir):
    """How many tensors the anchor actually covers, and how many --l2-sp-exclude left out."""
    pat = re.compile(r"(\d+) tensor\(s\) stay TRAINABLE but are OUT of the anchor, (\d+) anchored")
    with open(f"{rundir}/run.log", errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None, None


def trajectory(rundir, ep):
    rows = [r for r in csv.DictReader(open(f"{rundir}/trajectory.csv")) if r["epoch"] == str(ep)]
    gn = [float(r["grad_norm"]) for r in rows if r["grad_norm"] not in ("", "nan")]
    return {
        "steps": sum(1 for r in rows if r["step_taken"] == "1"),
        "max_grad_norm": max(gn) if gn else float("nan"),
        "nonfinite": sum(int(r["nonfinite_grad"]) for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rid", required=True)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--runs", default="training/runs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rundir = f"{a.runs}/{a.rid}"
    cache = RHO.load(a.runs, a.rid, a.budget)
    if not cache:
        sys.exit(f"no cached epoch for {a.rid} at budget {a.budget}")
    eps = sorted(cache)
    ep = eps[-1]
    row = dict(zip(RHO.COLS, cache[ep]))
    form, tau = RHO.run_rank_form(rundir)
    args = json.load(open(f"{rundir}/args.json"))
    # The launcher's --note is recorded in config.json, not args.json.
    cfg = {}
    try:
        cfg = json.load(open(f"{rundir}/config.json"))
    except Exception:
        pass
    tr = P3.train_rank_loss(a.rid)
    dr, vl = drift(rundir), valloss(rundir)
    ck, exn = excluded_norms(rundir)
    traj = trajectory(rundir, ep)
    tl = trainable_line(rundir)
    n_excl, n_anch = anchored_line(rundir)

    # SHIPPED, from this run's own shards: stored_iptm is the shipped head's ipTM per pose.
    rws = shards(rundir, ep)
    s_sel = _sel(rws, "stored_iptm")
    s_dq, s_dq_se = mean_sem(list(s_sel.values()))
    s_fr, s_fr_se = mean_sem([1.0 if v > 0.23 else 0.0 for v in s_sel.values()])
    oracle = {}
    for r in rws:
        k, d = r["iface"], float(r["dockq"])
        oracle[k] = max(oracle.get(k, 0.0), d)
    o_dq, _ = mean_sem(list(oracle.values()))
    o_fr, _ = mean_sem([1.0 if v > 0.23 else 0.0 for v in oracle.values()])

    L = []
    W = L.append
    note = (cfg.get("note") or args.get("note") or "").strip()
    title = note.split(".")[0].strip() if note else ""
    W(f"# {a.rid}: {title or 'arm'}\n")
    W(f"Regenerate with `python -I run_logging/render_findings.py --rid {a.rid}`. Every number is read")
    W(f"back from this run's own records; the shipped baselines are derived from the same eval shards")
    W(f"via their `stored_iptm` column, not carried over by hand.\n")
    W(f"**Epochs evaluated:** {eps[0]}-{ep} ({len(eps)} of {args.get('epochs')}).  "
      f"**Interfaces:** {row['n_ifaces']}.  **Budget:** {a.budget}.\n")

    W("## Configuration\n")
    W("| | |")
    W("|---|---|")
    # tau None in args means --rank-tau was not passed, so the FORM's default applied. Print the
    # value that was actually used rather than the word "default", which hides which number it was.
    if tau is None:
        eff = RKL.DEFAULT_SOFTMAX_TAU if form == "expected" else RKL.DEFAULT_TAU
        tau_txt = f", tau {eff} (the `{form}` default, --rank-tau not passed)"
    else:
        tau_txt = f", tau {tau}"
    W(f"| objective | `{form}`{tau_txt} |")
    W(f"| alpha_rank / alpha_pae | {args.get('alpha_rank')} / {args.get('alpha_pae')} |")
    if tl:
        W(f"| trainable (from the run's own log) | **{tl['tensors']} tensor(s), {tl['millions']}M "
          f"of {tl['of']}M parameters ({tl['pct']}%)** |")
    W(f"| `--train-only` patterns | {len((args.get('train_only') or '').split(',')) if args.get('train_only') else 'none (whole head)'} |")
    W(f"| L2-SP lambda | {args.get('l2_sp')} |")
    W(f"| L2-SP excluded | `{args.get('l2_sp_exclude') or '(none)'}`"
      + (f" -- {n_excl} tensor(s) out of the anchor, {n_anch} anchored" if n_anch else "") + " |")
    W(f"| pose injection | per-block={args.get('pose_perblock')} copies={args.get('pose_perblock_copy')} orientation={args.get('frame_direction')} |")
    W(f"| lr / warmup / epochs | {args.get('lr')} / {args.get('warmup_steps')} / {args.get('epochs')} |")
    W("")

    W(f"## Criterion: selected DockQ and fraction above 0.23\n")
    W("Spearman is reported but is **not** the criterion.\n")
    W("| | trained | shipped | oracle |")
    W("|---|---|---|---|")
    W(f"| selected DockQ | {row['dockq_mean']:.4f} +- {row['dockq_sem']:.4f} | {s_dq:.4f} +- {s_dq_se:.4f} | {o_dq:.4f} |")
    W(f"| frac > 0.23 | {row['frac_mean']:.4f} +- {row['frac_sem']:.4f} | {s_fr:.4f} +- {s_fr_se:.4f} | {o_fr:.4f} |")
    W(f"| Spearman rho | {row['rho_trained']:.4f} +- {row['rho_sem']:.4f} | {row['rho_shipped']:.4f} | |")
    W("")

    lab = "held-out ranking loss" if form != "expected" else "E[DockQ]"
    W(f"## Loss, per epoch\n")
    W(f"`{lab}` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for")
    W("comparability with the expected-DockQ arms and is **not** what this arm optimises.\n")
    W("| epoch | training | held-out ranking loss | E[DockQ] | drift \\|\\|W-W0\\|\\|^2 | penalty | val PAE |")
    W("|---|---|---|---|---|---|---|")
    for e in eps:
        r = dict(zip(RHO.COLS, cache[e]))
        d = dr.get(e, (float("nan"), float("nan")))
        W(f"| {e} | {tr.get(e, float('nan')):.5f} | {r['xsloss_trained']:.5f} +- {r['xsloss_sem']:.5f} "
          f"| {-r['exloss_trained']:.4f} | {d[0]:.1f} | {d[1]:.4f} | {vl.get(e, float('nan')):.4f} |")
    r0 = dict(zip(RHO.COLS, cache[ep]))
    W(f"\nShipped, on these same interfaces: held-out ranking loss {r0['xsloss_shipped']:.5f}, "
      f"E[DockQ] {-r0['exloss_shipped']:.4f} (tau {r0['exloss_tau']}).\n")

    W("## Selection metrics, per epoch\n")
    W("| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |")
    W("|---|---|---|---|---|")
    for e in eps:
        r = dict(zip(RHO.COLS, cache[e]))
        W(f"| {e} | {r['dockq_mean']:.4f} +- {r['dockq_sem']:.4f} | {r['frac_mean']:.4f} +- {r['frac_sem']:.4f} "
          f"| {r['rho_trained']:.4f} +- {r['rho_sem']:.4f} | {r['spread']:.4f} |")
    W("")

    if len(eps) > 1:
        p = paired(rundir, eps[0], ep)
        W(f"## Paired per-interface test, epoch {eps[0]} -> {ep}\n")
        W(f"Per-interface selected-pose DockQ (argmax ipTM), {p['n']} interfaces paired.\n")
        W(f"- paired mean **{p['mean']:+.4f} +- {p['sem']:.4f}** ({p['sigma']:.2f} sigma)")
        W(f"- {p['up']} up, {p['down']} worsened, {p['same']} unchanged")
        if p["movers"]:
            W("\n| interface | ep%d | ep%d | delta |" % (eps[0], ep))
            W("|---|---|---|---|")
            for k, va, vb, dd in p["movers"]:
                W(f"| `{k}` | {va:.3f} | {vb:.3f} | {dd:+.3f} |")
        W("")
        if eps[0] != 1:
            W(f"Note: epoch {eps[0]} is the earliest EVALUATED epoch, not a pre-training state.\n")

    if exn:
        W("## Tensors excluded from the anchor\n")
        W(f"`--l2-sp-exclude {args.get('l2_sp_exclude')}` leaves these trainable but outside")
        W("`||W - W0||^2`, so the drift column above does **not** cover them. They are zero-init, so")
        W(f"their norm is the direct test of whether the loss finds any use for them. From `{ck}`:\n")
        W("| tensor | \\|\\|W\\|\\| |")
        W("|---|---|")
        for k, v in sorted(exn.items()):
            W(f"| `{k.replace('confidence_head.','')}` | {v:.4e} |")
        W("")

    if note:
        W("## The arm as launched\n")
        W("Verbatim `--note` from `config.json`, written before any result was seen:\n")
        W("> " + note.replace("\n", " ") + "\n")
        if tl and "19.8M" in note and tl["millions"] < 19.0:
            W(f"**The note above is STALE on one point.** It describes all 248 head tensors / 19.8M")
            W(f"parameters, which is what was intended at the time it was written. `--grad-probe`")
            W(f"then REFUSED that set, because 158 of those tensors receive no ranking gradient at")
            W(f"all, and the arm was narrowed before launch to the {tl['tensors']} tensors /")
            W(f"{tl['millions']}M parameters the table above reports from the run's own log. See")
            W(f"`docs/confidence_head_reachability.md`. The note is left verbatim rather than edited,")
            W(f"because a run's recorded config should say what was configured, not be rewritten after")
            W(f"the fact -- but the number to trust is the logged one.\n")

    W("## Trajectory, latest epoch\n")
    W(f"{traj['steps']} optimiser step(s), max `grad_norm` {traj['max_grad_norm']:.4f}, "
      f"{traj['nonfinite']} non-finite gradient(s).\n")

    out = a.out or f"docs/{a.rid}_results.md"
    open(out, "w").write("\n".join(L) + "\n")
    print(f"wrote {out} ({len(L)} lines, epochs {eps[0]}-{ep})")


if __name__ == "__main__":
    main()
