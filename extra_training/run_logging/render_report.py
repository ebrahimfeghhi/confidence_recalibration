"""Render a training run's records as one self-contained HTML page: three charts and the config.

WHY A PAGE AND NOT A DASHBOARD SERVICE. The records are plain files (see records.py), so the viewer
is a free choice. A static page costs no account, no port forwarding and no server process, it can
be read by anyone the link is shared with, and it can be regenerated from the same records after the
run is over -- including for runs that already crashed. TensorBoard would need a tunnel to view and
writes binary event files that a reviewer reading the repo cannot open; for a run whose entire
history is 291 optimiser steps, that is heavy machinery. to_tensorboard.py exists for the case where
live curves over SSH are wanted anyway, and converts FROM these records rather than instrumenting
the trainer.

THE THREE CHARTS, and what a point is in each -- stated here and on the page, because a chart whose
axes are not named is not evidence:

  1. TRAINING LOSS PER BATCH. x = optimiser step. y = the summed confidence loss. One point is one
     optimiser step, which covers --accum poses (255: 85 per rank x 3 ranks), NOT one pose. A
     per-pose curve is 85x denser than the thing being optimised and reads as noise. The four terms
     are drawn separately because the total is their sum and a change in one is the interesting case.
  2. VALIDATION mean DockQ PER EPOCH. x = epoch. y = the mean over 92 interfaces of the DockQ of the
     pose the head was most confident in, over the full 100-seed budget. One point is one
     checkpoint scored on all 5,300 validation units.
  3. VALIDATION fraction acceptable PER EPOCH. Same axes; y = the fraction of those 92 interfaces
     whose selected pose reaches DockQ 0.23, the CAPRI acceptable threshold.

Both validation charts carry two reference lines: the SHIPPED head scored by this same evaluator
(what a trained head must beat) and the ORACLE, the best pose the sampler produced whether or not
confidence found it (the ceiling at zero extra compute). The gap between them is the headroom the
whole project is aimed at.
"""
import argparse, html, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import records as R

ACCEPTABLE = 0.23

# Keys worth showing without unfolding: the ones that change what the run IS.
KEY_ARGS = ["lr", "epochs", "accum", "alpha_confidence", "alpha_pae", "clip_norm", "warmup_steps",
            "dist_backend", "preload_items", "max_rss_gb", "eval_budget", "eval_every_epoch",
            "allow_code_drift", "seed", "shuffle", "fold", "labels", "out"]


def _axis(lo, hi, n=4):
    """Ticks that are round numbers inside [lo, hi], so every label names a value the chart reaches."""
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    import math
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), 10 * mag)
    first = math.ceil(lo / step) * step
    out, v = [], first
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def line_chart(series, *, xlab, ylab, width=760, height=260, refs=(), xint=False, note=""):
    """One SVG line chart. `series` is [(label, colour_token, [(x, y), ...])].

    Drawn by hand rather than with a library: three charts of a few hundred points each do not
    justify a dependency, and the viewBox has to leave room for the outermost labels, which generic
    wrappers tend not to do.
    """
    pad_l, pad_r, pad_t, pad_b = 58, 14, 14, 38
    pts = [p for _l, _c, s in series for p in s]
    ys = [y for _x, y in pts] + [r[0] for r in refs]
    xs = [x for x, _y in pts]
    if not ys:
        ys = [0.0, 1.0]
    if not xs:
        xs = [0.0, 1.0]
    ylo, yhi = min(ys), max(ys)
    if yhi - ylo < 1e-9:
        ylo, yhi = ylo - 0.5, yhi + 0.5
    ypad = (yhi - ylo) * 0.12
    ylo, yhi = ylo - ypad, yhi + ypad
    xlo, xhi = min(xs), max(xs)
    if xhi - xlo < 1e-9:
        xlo, xhi = xlo - 0.5, xhi + 0.5

    def X(v):
        return pad_l + (v - xlo) / (xhi - xlo) * (width - pad_l - pad_r)

    def Y(v):
        return height - pad_b - (v - ylo) / (yhi - ylo) * (height - pad_t - pad_b)

    o = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
         f'aria-label="{html.escape(ylab)} against {html.escape(xlab)}">']
    for t in _axis(ylo, yhi):
        o.append(f'<line class="grid" x1="{pad_l}" y1="{Y(t):.1f}" x2="{width-pad_r}" '
                 f'y2="{Y(t):.1f}"/>')
        o.append(f'<text class="tick" x="{pad_l-8}" y="{Y(t)+4:.1f}" text-anchor="end">'
                 f'{t:g}</text>')
    xticks = _axis(xlo, xhi, 5)
    if xint:
        xticks = [t for t in xticks if abs(t - round(t)) < 1e-9]
    for t in xticks:
        o.append(f'<text class="tick" x="{X(t):.1f}" y="{height-pad_b+18}" '
                 f'text-anchor="middle">{int(t) if xint else f"{t:g}"}</text>')
    o.append(f'<line class="axis" x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" '
             f'y2="{height-pad_b}"/>')
    for val, lab, tok in refs:
        o.append(f'<line class="ref ref-{tok}" x1="{pad_l}" y1="{Y(val):.1f}" '
                 f'x2="{width-pad_r}" y2="{Y(val):.1f}"/>')
        o.append(f'<text class="reflab ref-{tok}" x="{width-pad_r-4}" y="{Y(val)-6:.1f}" '
                 f'text-anchor="end">{html.escape(lab)} {val:.4f}</text>')
    for lab, tok, s in series:
        if not s:
            continue
        d = " ".join(("M" if i == 0 else "L") + f"{X(x):.1f},{Y(y):.1f}"
                     for i, (x, y) in enumerate(s))
        o.append(f'<path class="ln ln-{tok}" d="{d}"/>')
        if len(s) <= 24:
            for x, y in s:
                o.append(f'<circle class="dot dot-{tok}" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5"/>')
    o.append(f'<text class="axlab" x="{(pad_l+width-pad_r)/2:.0f}" y="{height-4}" '
             f'text-anchor="middle">{html.escape(xlab)}</text>')
    o.append(f'<text class="axlab" transform="rotate(-90 14 {(height-pad_b)/2:.0f})" x="14" '
             f'y="{(height-pad_b)/2:.0f}" text-anchor="middle">{html.escape(ylab)}</text>')
    if not pts:
        o.append(f'<text class="empty" x="{(pad_l+width-pad_r)/2:.0f}" '
                 f'y="{(height-pad_b+pad_t)/2:.0f}" text-anchor="middle">'
                 f'{html.escape(note or "no data yet")}</text>')
    o.append("</svg>")
    return "\n".join(o)


def build(run, summ, generated, gen_epoch=0):
    a, base = run["args"], run["baseline"]
    cfg, stat = run.get("config") or {}, run.get("status") or {}
    # THE EFFECTIVE BATCH, not the requested one: 256 over three ranks is 85 each and 255 in total.
    eff, local, world = R.effective_batch(run)
    batch = eff if eff else a.get("accum", "?")
    # THE COMMIT AND THE REASON FOR THE LAUNCH, on the page rather than only in config.json. A curve
    # cannot be compared with another unless the code behind it is identifiable, and a run that was
    # a restart should say what it was a restart FROM.
    prov = ""
    if cfg:
        bits = [f'run <code>{html.escape(str(cfg.get("run_id", "?")))}</code>',
                f'commit <code>{html.escape(str(cfg.get("commit", "?"))[:8])}</code>'
                + (" <strong>(dirty)</strong>" if cfg.get("dirty") else "")]
        if cfg.get("cards"):
            bits.append(f'cards <code>{html.escape(str(cfg["cards"]))}</code>')
        if stat.get("exit_code") is not None:
            bits.append(f'exited <code>{stat["exit_code"]}</code> after {stat.get("wall_min","?")} min')
        prov = " &middot; ".join(bits)
        if cfg.get("note"):
            prov += f'<br>{html.escape(str(cfg["note"]))}'
    elif summ["poses"]:
        prov = ("No <code>config.json</code>: this run predates "
                "<code>run_logging/launch.py</code>, so its commit is not recorded.")
    steps, eps = run["steps"], run["epochs"]

    # ---- chart 1: the per-batch loss
    total = [(p["step"], p["raw"]) for p in steps]
    comps = [("pLDDT", "a", [(p["step"], p["plddt_loss"]) for p in steps if p["plddt_loss"] is not None]),
             ("PAE", "b", [(p["step"], p["pae_loss"]) for p in steps if p["pae_loss"] is not None]),
             ("PDE", "c", [(p["step"], p["pde_loss"]) for p in steps if p["pde_loss"] is not None]),
             ("resolved", "d", [(p["step"], p["resolved_loss"]) for p in steps if p["resolved_loss"] is not None])]
    c1 = line_chart([("total", "t", total)] + comps,
                    xlab=f"optimiser step  (one step = {batch} poses)",
                    ylab="summed confidence loss",
                    note="waiting for the first optimiser step")

    # ---- charts 2 and 3: validation, per epoch
    def refs(k, ok):
        r = []
        if base.get(k) is not None:
            r.append((base[k], "shipped head", "base"))
        if base.get(ok) is not None:
            r.append((base[ok], "oracle", "orc"))
        return r
    c2 = line_chart([("trained head", "t", [(m["epoch"], m["mean_dockq"]) for m in eps
                                            if m.get("mean_dockq") is not None])],
                    xlab="epoch", ylab="mean DockQ of the selected pose", xint=True,
                    refs=refs("mean_dockq", "oracle_mean_dockq"),
                    note="pending: the first value lands at the end of epoch 1")
    c3 = line_chart([("trained head", "t", [(m["epoch"], m["frac_acc"]) for m in eps
                                            if m.get("frac_acc") is not None])],
                    xlab="epoch", ylab=f"fraction with DockQ ≥ {ACCEPTABLE}", xint=True,
                    refs=refs("frac_acc", "oracle_frac_acc"),
                    note="pending: the first value lands at the end of epoch 1")

    def row(k, v):
        return (f'<div class="kv"><dt>{html.escape(str(k))}</dt>'
                f'<dd>{html.escape(str(v))}</dd></div>')
    shown = "".join(row(k, a[k]) for k in KEY_ARGS if k in a)
    rest = "".join(row(k, a[k]) for k in sorted(a) if k not in KEY_ARGS)

    ep_rows = "".join(
        f"<tr><td>{m['epoch']}</td><td>{m.get('mean_dockq', '')}</td>"
        f"<td>{m.get('frac_acc', '')}</td><td>{m.get('oracle_mean_dockq', '')}</td>"
        f"<td>{m.get('headroom', '')}</td>"
        f"<td>{html.escape(str(m.get('ckpt', '')))}</td></tr>" for m in eps)
    if not ep_rows:
        ep_rows = ('<tr><td colspan="6" class="pending">no epoch has finished yet; the shipped-head '
                   'reference is in the charts above</td></tr>')

    warn = ""
    if run["warnings"]:
        warn = ('<div class="warn"><strong>Reading the records raised:</strong><ul>'
                + "".join(f"<li>{html.escape(w)}</li>" for w in run["warnings"]) + "</ul></div>")

    beat = base.get("mean_dockq")
    beat_s = f"{beat:.4f}" if beat is not None else "unknown"
    gm = f"{summ['grad_med']:.2f}" if "grad_med" in summ else "—"
    gx = f"{summ['grad_max']:.2f}" if "grad_max" in summ else "—"
    rawf = f"{summ['raw_first']:.3f}" if "raw_first" in summ else "—"
    rawl = f"{summ['raw_last']:.3f}" if "raw_last" in summ else "—"

    return f"""<title>FT3 Confidence Finetune</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one column, summary strip first, then the three charts stacked, config last. */
:root {{
  --bg:#f4f5f7; --panel:#ffffff; --fg:#1b2027; --mute:#5d6673; --line:#dde1e7;
  --accent:#b4690e; --ref:#2a7f7a; --ok:#2f6b3a; --warn:#8a5a12;
  --sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  color-scheme:light;
}}
@media (prefers-color-scheme:dark) {{ :root:not([data-theme="light"]) {{
  --bg:#14171c; --panel:#1b1f26; --fg:#e6e9ee; --mute:#9aa4b2; --line:#2b313a;
  --accent:#e0a040; --ref:#5bb8b0; --ok:#7bc08a; --warn:#d8a44a; color-scheme:dark;
}} }}
:root[data-theme="dark"] {{
  --bg:#14171c; --panel:#1b1f26; --fg:#e6e9ee; --mute:#9aa4b2; --line:#2b313a;
  --accent:#e0a040; --ref:#5bb8b0; --ok:#7bc08a; --warn:#d8a44a; color-scheme:dark;
}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--fg);font-family:var(--sans);margin:0}}
.wrap{{max-width:860px;margin:0 auto;padding-block:28px;padding-left:16px;padding-right:16px}}
h1{{font-size:1.4rem;font-weight:600;margin:0 0 2px;letter-spacing:-.01em;text-wrap:balance}}
.sub{{color:var(--mute);font-size:.84rem;margin:0 0 22px}}
.sub code{{font-family:var(--mono);font-size:.82rem}}
.strip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:6px;overflow:hidden;
  margin-bottom:24px}}
.cell{{background:var(--panel);padding:11px 13px;min-width:0}}
.cell .lab{{font-size:.64rem;text-transform:uppercase;letter-spacing:.07em;color:var(--mute)}}
.cell .val{{font-family:var(--mono);font-size:1.02rem;font-variant-numeric:tabular-nums;
  margin-top:3px}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:16px;
  margin-bottom:18px}}
h2{{font-size:.95rem;font-weight:600;margin:0 0 3px}}
.axes{{color:var(--mute);font-size:.76rem;margin:0 0 12px;line-height:1.5}}
.chartbox{{overflow-x:auto}}
.chart{{width:100%;min-width:460px;height:auto;display:block}}
.grid{{stroke:var(--line);stroke-width:1}}
.axis{{stroke:var(--line);stroke-width:1.5}}
.tick,.axlab{{fill:var(--mute);font-family:var(--mono);font-size:10px}}
.axlab{{font-size:10.5px}}
.empty{{fill:var(--mute);font-family:var(--sans);font-size:12px;font-style:italic}}
.ln{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
.ln-t{{stroke:var(--accent);stroke-width:2.4}}
.ln-a{{stroke:var(--mute);stroke-width:1.1;opacity:.75}}
.ln-b{{stroke:var(--ref);stroke-width:1.4;opacity:.9}}
.ln-c{{stroke:var(--mute);stroke-width:1.1;opacity:.5;stroke-dasharray:4 3}}
.ln-d{{stroke:var(--mute);stroke-width:1.1;opacity:.35;stroke-dasharray:2 3}}
.dot{{fill:var(--accent);stroke:var(--panel);stroke-width:1.2}}
.ref{{stroke-width:1.3;stroke-dasharray:6 4}}
.ref-base{{stroke:var(--ok)}} .ref-orc{{stroke:var(--ref)}}
.reflab{{font-family:var(--mono);font-size:9.5px}}
.reflab.ref-base{{fill:var(--ok)}} .reflab.ref-orc{{fill:var(--ref)}}
.key{{display:flex;flex-wrap:wrap;gap:12px;margin-top:10px;font-size:.73rem;color:var(--mute);
  font-family:var(--mono)}}
.key i{{display:inline-block;width:16px;height:0;border-top:2px solid currentColor;
  vertical-align:middle;margin-right:5px}}
.k-t{{color:var(--accent)}} .k-b{{color:var(--ref)}} .k-base{{color:var(--ok)}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums}}
th{{font-size:.66rem;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);
  font-weight:500}}
td{{font-family:var(--mono);font-size:.8rem}}
.pending{{color:var(--mute);font-style:italic;font-family:var(--sans)}}
.kvs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:0 18px;margin:0}}
.kv{{display:flex;justify-content:space-between;gap:10px;padding:5px 0;
  border-bottom:1px solid var(--line);min-width:0}}
.kv dt{{color:var(--mute);font-size:.76rem}}
.kv dd{{margin:0;font-family:var(--mono);font-size:.76rem;overflow-wrap:anywhere;text-align:right}}
details{{margin-top:12px}}
summary{{cursor:pointer;font-size:.78rem;color:var(--mute)}}
summary:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
.warn{{background:var(--panel);border:1px solid var(--warn);border-left-width:3px;
  border-radius:6px;padding:12px 14px;margin-bottom:18px;font-size:.8rem}}
.warn ul{{margin:6px 0 0;padding-left:18px}}
.foot{{color:var(--mute);font-size:.73rem;margin-top:22px;line-height:1.6}}
</style>

<div class="wrap">
  <h1>FT3 Confidence Finetune</h1>
  <p class="sub">Confidence head only, 19.8M of 464.4M parameters, trunk frozen. Target to beat:
    <code>{beat_s}</code> mean DockQ, the shipped head scored by this same evaluator.
    Generated {html.escape(generated)} from <code>{html.escape(os.path.basename(run['out_dir']))}</code>,
    <span id="age" data-gen="{gen_epoch}">just now</span>.</p>
  {f'<p class="sub" style="margin-top:-14px">{prov}</p>' if prov else ''}

  {warn}

  <div class="strip">
    <div class="cell"><div class="lab">epoch</div><div class="val">{summ['epoch']} / {summ.get('epochs_total','?')}</div></div>
    <div class="cell"><div class="lab">steps</div><div class="val">{summ['steps']}</div></div>
    <div class="cell"><div class="lab">poses</div><div class="val">{summ['poses']}</div></div>
    <div class="cell"><div class="lab">wall</div><div class="val">{summ['wall_min']:.0f} min</div></div>
    <div class="cell"><div class="lab">lr</div><div class="val">{summ.get('lr','?')}</div></div>
    <div class="cell"><div class="lab">grad norm</div><div class="val">{gm}</div></div>
    <div class="cell"><div class="lab">dropped steps</div><div class="val">{summ['dropped']}</div></div>
    <div class="cell"><div class="lab">val points</div><div class="val">{summ['evals']}</div></div>
  </div>

  <section>
    <h2>Training loss per batch</h2>
    <p class="axes">x: optimiser step &mdash; one step is <strong>{batch}</strong> poses
      ({local} per rank &times; {world} ranks; <code>--accum {a.get('accum','?')}</code> is the
      request, and {a.get('accum','?')}&nbsp;&divide;&nbsp;{world} does not divide evenly).
      y: the confidence loss, as the sum of
      its four terms. One point is one step, not one pose. Raw terms, unweighted
      (<code>alpha_confidence = {a.get('alpha_confidence','?')}</code>).</p>
    <div class="chartbox">{c1}</div>
    <div class="key"><span class="k-t"><i></i>total</span><span class="k-b"><i></i>PAE</span>
      <span><i></i>pLDDT</span><span><i></i>PDE (dashed)</span><span><i></i>resolved (dotted)</span></div>
    <p class="axes" style="margin-top:10px;margin-bottom:0">PAE is the term that matters: ipTM, which
      selects poses, is computed entirely from PAE. Protenix ships its weight at 0, so this run sets
      <code>alpha_pae = {a.get('alpha_pae','?')}</code>.</p>
  </section>

  <section>
    <h2>Validation &mdash; mean DockQ of the selected pose</h2>
    <p class="axes">x: epoch. y: mean over {base.get('n_interfaces','92')} interfaces of the DockQ of
      the pose the head ranked highest by per-interface ipTM, over the full
      {base.get('budget','100')}-seed budget. One point is one checkpoint scored on all 5,300
      validation units.</p>
    <div class="chartbox">{c2}</div>
    <div class="key"><span class="k-t"><i></i>trained head</span>
      <span class="k-base"><i></i>shipped head</span><span class="k-b"><i></i>oracle (best pose produced)</span></div>
  </section>

  <section>
    <h2>Validation &mdash; fraction acceptable</h2>
    <p class="axes">x: epoch. y: the fraction of those interfaces whose selected pose reaches
      DockQ&nbsp;&ge;&nbsp;{ACCEPTABLE}, the CAPRI acceptable threshold. Same selection rule as above.</p>
    <div class="chartbox">{c3}</div>
    <div class="key"><span class="k-t"><i></i>trained head</span>
      <span class="k-base"><i></i>shipped head</span><span class="k-b"><i></i>oracle</span></div>
  </section>

  <section>
    <h2>Validation by epoch</h2>
    <table><thead><tr><th>epoch</th><th>mean dockq</th><th>frac acc</th><th>oracle</th>
      <th>headroom</th><th>checkpoint</th></tr></thead><tbody>{ep_rows}</tbody></table>
  </section>

  <section>
    <h2>Run configuration</h2>
    <dl class="kvs">{shown}</dl>
    <details><summary>every other argument ({len(a) - sum(1 for k in KEY_ARGS if k in a)})</summary>
      <dl class="kvs">{rest}</dl></details>
  </section>

  <script>
  // How old is this page? The records are re-rendered every 5 minutes on the machine, but the
  // published page changes only when it is republished, so the gap is shown rather than hidden.
  (function () {{
    var el = document.getElementById("age");
    if (!el) return;
    var gen = parseInt(el.getAttribute("data-gen"), 10) * 1000;
    function tick() {{
      if (!gen) return;
      var mins = Math.floor((Date.now() - gen) / 60000);
      el.textContent = mins < 1 ? "less than a minute ago"
        : mins === 1 ? "1 minute ago"
        : mins < 90 ? mins + " minutes ago"
        : Math.round(mins / 60) + " hours ago";
      el.style.color = mins > 20 ? "var(--warn)" : "";
    }}
    tick();
    setInterval(tick, 30000);
  }})();
  </script>

  <p class="foot">Records: <code>trajectory.csv</code> (one row per pose),
    <code>metric_ep&lt;N&gt;.json</code> (one per epoch), <code>args.json</code>. Rendered by
    <code>run_logging/render_report.py</code>; re-run it to refresh. The trainer carries no viewer
    instrumentation, so the records outlive whatever reads them.</p>
</div>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="a run's --out directory")
    ap.add_argument("--run-id", default="",
                    help="a run id under training/runs/, or `latest`. Resolves to its directory, so "
                         "a report can be asked for by the id the index lists.")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--out", required=True, help="where to write the html")
    a = ap.parse_args()
    import datetime
    if a.run_id:
        a.run = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "training", "runs", a.run_id)
    if not a.run:
        sys.exit("STOP: pass --run <dir> or --run-id <id>")
    if not os.path.isdir(a.run):
        sys.exit(f"STOP: {a.run} is not a directory")
    run = R.read_run(a.run, a.baseline)
    summ = R.summarise(run)
    gen = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    # The epoch seconds go in too, so the page can say how stale it is. The FILE is re-rendered
    # every 5 min by watch.sh, but the PUBLISHED page only changes when it is republished, so a
    # reader needs to see the gap rather than assume there is none.
    gen_epoch = int(datetime.datetime.now().timestamp())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(build(run, summ, gen, gen_epoch))
    print(f"{a.out}: epoch {summ['epoch']}, {summ['steps']} step(s), {summ['poses']} pose(s), "
          f"{summ['evals']} val point(s)")
    for w in run["warnings"]:
        print(f"  note: {w}")


if __name__ == "__main__":
    main()
