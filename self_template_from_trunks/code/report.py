#!/usr/bin/env python
"""30-minute health + scoring report for the self_template_from_trunks run. Read-only. Prints plain text."""
import collections, csv, glob, json, os, re, statistics, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as pl

E = pl.EXP; roles, tok = pl.load_info(); order = sorted((t, p) for p, t in tok.items() if p not in pl.DONE_ALREADY); order = [p for _, p in order]
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
now = time.strftime('%Y-%m-%d %H:%M'); print(f'=== self_template_from_trunks report {now} ===')
# --- health
ctl = sh("tmux ls 2>/dev/null | grep -E '^st_prod' | cut -d: -f1"); pids = sh("pgrep -f '^/home/ebrahim/Protenix/.venv/bin/python pipeline.py run' | head -1")
print(f'controller: {"ALIVE (tmux " + ctl + ", pid " + pids + ")" if pids else "NOT RUNNING"}')
st = {p: pl.status(p, roles) for p in order}; c = collections.Counter(st.values())
print('complexes: ' + ', '.join(f'{k} {v}' for k, v in sorted(c.items())) + f'  (total {len(order)})')
infl = [(p, st[p]) for p in order if st[p] in ('A', 'B', 'C', 'D')]; gb = sum(pl.trunk_gb(tok[p]) for p, _ in infl)
print(f'in flight: {", ".join(f"{p}:{s}" for p, s in infl) or "none"} | trunk estimate {gb:.0f} GB of the 450 GB cap')
line = sh(f"grep ' status ' {E}/controller.log | tail -1"); print('last controller status: ' + (line[line.find('status'):] if line else 'n/a'))
print('GPUs: ' + '; '.join(sh(f"nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i {g}").replace(', ', ' ') for g in (0, 1, 2, 5)))
holders = sh(f"grep -c 'holder started' {E}/controller.log"); print(f'GPU holders started so far: {holders} (idle-card events)')
oom = sh(f"cat {E}/data/*/gpu_*.log 2>/dev/null | grep -ci 'out of memory\\|OutOfMemory'"); fail = sh(f"cat {E}/data/*/gpu_*.log 2>/dev/null | grep -c 'FAILED'")
print(f'errors in fold logs: out-of-memory {oom}, FAILED units {fail}')
parked = [(p, open(f'{pl.R(p)}/PARKED.txt').read().strip()[:160]) for p in order if st[p] == 'parked']
print('PARKED: ' + ('none' if not parked else '\n  ' + '\n  '.join(f'{p}: {w}' for p, w in parked)))
q = sh("quota -s 2>&1 | tail -1").split(); print(f'quota: {q[1]} used of {q[3]} limit' if len(q) > 3 else 'quota: n/a')
# --- scoring
sc = E + '/summary_all.csv'; rows = list(csv.DictReader(open(sc))) if os.path.exists(sc) else []
mark = E + '/.report_marker'; seen = int(open(mark).read()) if os.path.exists(mark) else 0
if rows:
    f = sum(int(r['free_acceptable']) for r in rows); k = sum(int(r['ctl_acceptable']) for r in rows); t = sum(int(r['arm_acceptable']) for r in rows); n = len(rows)
    print(f'\nSCORING, {len({r["pdb"] for r in rows})} complexes / {n} interfaces finished. Acceptable (top-ipTM pick, DockQ>=0.23): free {f}/{n}, control {k}/{n}, templated {t}/{n}')
    # class from the free run's own 250 poses: solved = free top-ipTM pick acceptable; decoy = pick wrong but an acceptable pose exists in the 250; no_acceptable = none
    for r in rows:
        dq = [float(x['dockq']) for x in pl.read_rows(f'{pl.R(r["pdb"])}/free/results.csv') if x['iface'] == r['interface']]
        r['pool_best'] = max(dq) if dq else 0.0
        r['cls'] = 'solved' if int(r['free_acceptable']) else ('decoy' if r['pool_best'] >= 0.23 else 'no_acceptable')
    with open(E + '/summary_classes.csv', 'w', newline='') as fh:   # derived, rewritten every report
        w = csv.writer(fh); w.writerow(['pdb', 'interface', 'class', 'donors', 'pool_best_dockq', 'free_pick_dockq', 'ctl_pick_dockq', 'arm_pick_dockq', 'free_ok', 'ctl_ok', 'arm_ok'])
        for r in rows: w.writerow([r['pdb'], r['interface'], r['cls'], r['donors'], round(r['pool_best'], 4), r['free_pick_dockq'], r['ctl_pick_dockq'], r['arm_pick_dockq'], r['free_acceptable'], r['ctl_acceptable'], r['arm_acceptable']])
    for cls, what in (('solved', 'kept (still acceptable)'), ('decoy', 'RESCUED (pick now acceptable)'), ('no_acceptable', 'found an acceptable pick')):
        g = [r for r in rows if r['cls'] == cls]
        print(f'  {cls:14s} {len(g):3d} interfaces | control {what}: {sum(int(r["ctl_acceptable"]) for r in g)}, templated {what}: {sum(int(r["arm_acceptable"]) for r in g)}')
    print(f'  health metrics (median |ipTM change| vs the same seed\'s free fold): control {statistics.median(float(r["ctl_med_abs_diptm_vs_free"]) for r in rows):.3f}, templated {statistics.median(float(r["arm_med_abs_diptm_vs_free"]) for r in rows):.3f}')
    new = rows[seen:]
    if new:
        print(f'  new since last report ({len(new)} interfaces): free -> control -> templated DockQ of the top-ipTM pick')
        for r in new: print(f'    {r["interface"]:9s} [{r["cls"]}] donors {r["donors"]:>3s} | free {float(r["free_pick_dockq"]):.2f}{"*" if r["free_acceptable"]=="1" else " "} | ctl {float(r["ctl_pick_dockq"]):.2f}{"*" if r["ctl_acceptable"]=="1" else " "} | templated {float(r["arm_pick_dockq"]):.2f}{"*" if r["arm_acceptable"]=="1" else " "}')
    open(mark, 'w').write(str(len(rows)))
else: print('\nSCORING: no complex finished yet')
# --- ETA (cost-weighted by tokens^1.6 as a rough proxy for fold time; refined by measured progress)
w = lambda p: tok[p] ** 1.6; tot = sum(w(p) for p in order); dn = sum(w(p) for p in order if st[p] == 'done') + sum(0.5 * w(p) for p in order if st[p] in ('A', 'B', 'C', 'D'))
st0 = os.path.getmtime(E + '/controller.log') if False else None
t0f = E + '/production_started_epoch.txt'
if os.path.exists(t0f) and dn > 0:
    t0 = float(open(t0f).read()); el = time.time() - t0
    eta = el * (tot / dn - 1); print(f'\nrough ETA: {eta / 3600:.1f} h more (elapsed {el / 3600:.1f} h; token-weighted, coarse until several complexes finish)')
