#!/usr/bin/env python
"""Keep the baseline and method arms finishing TOGETHER by moving GPUs between them.

    ./rebalance_gpus.py                 # report the decision, change nothing
    ./rebalance_gpus.py --apply         # make at most ONE move
    ./rebalance_gpus.py --apply --loop  # every INTERVAL seconds (default 900)

WHY. An interface enters the comparison only when BOTH arms have finished its complex, so an arm that
races ahead buys nothing -- its folds sit waiting on the other arm. A fixed split cannot stay right:
the method began with 28.5 GPU-hours against the baseline's 21.2, and as each drains the ratio moves.
This equalises PREDICTED FINISH TIME, not fold counts, because the arms' remaining work differs in
both size and per-fold cost.

HOW THE ESTIMATE WORKS. Remaining folds per arm come from (queue - results.csv), and each is costed at
that complex's own measured median fold time where there are at least 3 observations, else the arm's
median seconds-per-1000-atoms times its atom count. Measured on 4,500+ folds the two arms cost the
same per atom (9.6 vs 9.8 s/1000), which is what makes the compute match real.

SAFETY RULES, each from something that has actually gone wrong here:
  at most 4 GPUs      the hard ceiling for this machine; never exceeded, never inferred from idle cards
  one move per tick   moving two at once thrashes and costs a model load (~1 min) each time
  a 25% / 1 h margin  no move unless the imbalance is worth the restart
  never leave an arm empty while it still has work
  WAIT FOR THE PID    a killed worker's claim is released only once its process is really gone. On
                      2026-10-06 a reconcile ran while the old pid was still exiting, claim_is_live
                      spared three claims on 8fbw, and that complex could never complete -- the
                      comparison's decoy row sat frozen for an hour with every GPU at 100%.
  reconcile after     then release the stale claims so the folds are refoldable
  holders stay        hold<GPU> is a separate session and is left alone; launch.sh re-creates it
"""
import argparse, collections, csv, glob, json, os, statistics as st, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/analysis"); sys.path.insert(0, "/home/ebrahim/tts_abag")
FB = "/home/ebrahim/tts_abag/experimental_data/protenix/foldbench"
ARMS = {
    "free":   {"dir": f"{FB}/free",                  "prefix": "free",  "poses": ""},
    "method": {"dir": f"{FB}/selftmpl_mask_on_nc10", "prefix": "nc10_", "poses": "poses"},
}
# REBALANCE_GPUS=<comma list> narrows the pool, so scaling the experiment down to fewer cards is one
# env var rather than an edit. The 4-card ceiling for this machine still applies as an upper bound.
GPUS = [int(g) for g in os.environ.get("REBALANCE_GPUS", "0,1,2,3").split(",") if g.strip()]
LOG = f"{FB}/rebalance.log"
SIZES = f"{FB}/selftmpl_mask_on_nc10/sizes.json"


def say(m):
    line = f"{time.strftime('%F %T')} {m}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def sessions(prefix):
    out = sh("tmux ls 2>/dev/null") or ""
    return sorted(l.split(":")[0] for l in out.splitlines()
                  if l.split(":")[0].startswith(prefix)
                  and l.split(":")[0][len(prefix):].isdigit())


def gpu_of(sess, prefix):
    return int(sess[len(prefix):])


def remaining(arm):
    a = ARMS[arm]
    q, sc = collections.defaultdict(set), collections.defaultdict(set)
    for l in open(f"{a['dir']}/queue.tsv"):
        f = l.rstrip("\n").split("\t")
        q[f[1]].add((int(f[4]), f[3]))
    if os.path.exists(f"{a['dir']}/results.csv"):
        for r in csv.DictReader(open(f"{a['dir']}/results.csv")):
            if r.get("dockq"):
                sc[r["iface"].split("_")[0]].add((int(r["seed"]), r["arm"]))
    return {p: len(q[p] - sc[p]) for p in q}


def cost(arm, sizes):
    """out: GPU-hours of work left in this arm"""
    a = ARMS[arm]
    pat = f"{a['dir']}/{a['poses']}/*/*/run.json" if a["poses"] else f"{a['dir']}/*/*/run.json"
    obs, rates = collections.defaultdict(list), []
    for f in glob.glob(pat):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        s, p = j.get("fold_seconds"), j.get("pdb")
        if s and p:
            obs[p].append(s)
            if sizes.get(p):
                rates.append(s / (sizes[p] / 1000))
    rate = st.median(rates) if rates else 10.0
    sec = 0.0
    for p, n in remaining(arm).items():
        if not n:
            continue
        per = st.median(obs[p]) if len(obs.get(p, [])) >= 3 else rate * sizes.get(p, 9000) / 1000
        sec += n * per
    return sec / 3600


def migrate(frm, to, gpu, dataset="foldbench"):
    """Stop frm's worker on `gpu`, release what it stranded, start to's worker there."""
    sess = f"{ARMS[frm]['prefix']}{gpu}"
    pane = sh(f"tmux list-panes -t {sess} -F '#{{pane_pid}}' 2>/dev/null")
    child = sh(f"pgrep -P {pane} 2>/dev/null | head -1") if pane else ""
    sh(f"tmux kill-session -t {sess}")
    if child:
        # the claim of an in-flight fold is only releasable once this pid is really gone
        for _ in range(150):
            if sh(f"kill -0 {child} 2>/dev/null; echo $?") != "0":
                break
            time.sleep(2)
    out = sh(f"{sys.executable} {HERE}/reconcile_stale_claims.py {ARMS[frm]['dir']} --dataset {dataset}")
    say(f"  moved GPU {gpu}: {frm} -> {to}  ({out})")
    sh(f"tmux new-session -d -s {ARMS[to]['prefix']}{gpu} "
       f"\"bash {ARMS[to]['dir']}/launch.sh {gpu}\"")


def revive(apply_):
    """Restart a worker that died, and claim an idle GPU if an arm has work and no card.

    The rebalancer OWNS gpu assignment now, so the watchers must run with GPUS='' -- two things
    deciding what belongs on a card is how a duplicate worker lands on a GPU that already has one.
    That means worker revival has to live here too, since that is what the watchers used to do.
    """
    have = {a: sessions(ARMS[a]["prefix"]) for a in ARMS}
    used = {gpu_of(s, ARMS[a]["prefix"]) for a in ARMS for s in have[a]}
    idle = [g for g in GPUS if g not in used]
    sizes = json.load(open(SIZES))
    work = {a: cost(a, sizes) for a in ARMS}
    # FIRST: an arm with work and no worker at all -- that is a stall, not an imbalance.
    for a in ARMS:
        if work[a] > 0.05 and not have[a] and idle:
            g = idle.pop(0)
            say(f"  REVIVE: {a} has work and no worker -- starting on GPU {g}")
            if apply_:
                sh(f"tmux new-session -d -s {ARMS[a]['prefix']}{g} "
                   f"\"bash {ARMS[a]['dir']}/launch.sh {g}\"")
            have[a] = have[a] + [f"{ARMS[a]['prefix']}{g}"]
    # THEN: any card left idle goes to whichever arm would finish LAST. A card borrowed for a one-off
    # job and handed back sat idle for an hour because revival only covered the empty-arm case, while
    # the migrate path refused to help -- moving free -> method would have made free the laggard, so
    # it correctly declined, and nothing claimed the free card. An idle GPU is always worth taking.
    while idle:
        cand = [a for a in ARMS if work[a] > 0.05]
        if not cand:
            break
        a = max(cand, key=lambda x: work[x] / max(len(have[x]), 1))
        g = idle.pop(0)
        say(f"  CLAIM IDLE: GPU {g} -> {a} (its ETA "
            f"{work[a] / max(len(have[a]), 1):.1f} h is the worst)")
        if apply_:
            sh(f"tmux new-session -d -s {ARMS[a]['prefix']}{g} "
               f"\"bash {ARMS[a]['dir']}/launch.sh {g}\"")
        have[a] = have[a] + [f"{ARMS[a]['prefix']}{g}"]


def tick(apply_):
    sizes = json.load(open(SIZES))
    revive(apply_)
    have = {a: sessions(ARMS[a]["prefix"]) for a in ARMS}
    work = {a: cost(a, sizes) for a in ARMS}
    n = {a: len(have[a]) for a in ARMS}
    eta = {a: (work[a] / n[a] if n[a] else float("inf")) for a in ARMS}
    say(f"free {work['free']:.1f} GPU-h on {n['free']} gpu -> {eta['free']:.1f} h | "
        f"method {work['method']:.1f} GPU-h on {n['method']} gpu -> {eta['method']:.1f} h")
    total = sum(n.values())
    if total > len(GPUS):
        say(f"  REFUSING: {total} workers exceeds the {len(GPUS)}-GPU ceiling")
        return
    done = [a for a in ARMS if work[a] < 0.05]
    for a in done:
        if n[a] and any(work[b] > 0.05 for b in ARMS if b != a):
            other = [b for b in ARMS if b != a][0]
            if apply_:
                migrate(a, other, gpu_of(have[a][0], ARMS[a]["prefix"]))
            else:
                say(f"  would move a GPU: {a} has no work left")
            return
    if any(work[a] < 0.05 for a in ARMS):
        return
    ahead = min(ARMS, key=lambda a: eta[a])     # finishes soonest -> can give one up
    behind = max(ARMS, key=lambda a: eta[a])
    if ahead == behind or n[ahead] <= 1:
        say("  no move: cannot free a GPU without emptying an arm")
        return
    gap = eta[behind] - eta[ahead]
    if gap < 1.0 or gap / max(eta[behind], 1e-9) < 0.25:
        say(f"  no move: gap {gap:.1f} h is within the margin")
        return
    # would the move actually help? compare the worse finish time before and after
    after = max(work[ahead] / (n[ahead] - 1), work[behind] / (n[behind] + 1))
    if after >= max(eta.values()) - 0.1:
        say(f"  no move: it would not improve the finish ({after:.1f} h vs {max(eta.values()):.1f} h)")
        return
    say(f"  gap {gap:.1f} h; moving one GPU {ahead} -> {behind} "
        f"(worst finish {max(eta.values()):.1f} h -> {after:.1f} h)")
    if apply_:
        migrate(ahead, behind, gpu_of(have[ahead][0], ARMS[ahead]["prefix"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=900)
    A = ap.parse_args()
    while True:
        try:
            tick(A.apply)
        except Exception as e:
            say(f"  ERROR in tick: {type(e).__name__}: {e}")
        if not A.loop:
            return
        time.sleep(A.interval)


if __name__ == "__main__":
    main()
