#!/usr/bin/env python
"""Choose --ckpt-token-threshold by MEASURING peak memory and backward time, never by arithmetic.

THE QUESTION. The arm runs --blocks-per-ckpt 1 -- every one of the head's four pairformer blocks
recomputed in backward -- on EVERY unit, because the largest ones need it. gpu_backward is 70.3
min of a 143.9 min epoch (epoch 17, CUDA events, so this is device time and not a launch timer),
and roughly 30% of a checkpointed backward is recompute. The 400-token units are paying that for
nothing.

WHY A THRESHOLD IS SAFE AT ALL. The peak is a PER-POSE property: _one_pose is a function, its
frame is destroyed on return, and every large tensor is freed before the next pose allocates. So
an uncheckpointed small unit does not ADD to the peak a large unit sets -- it only has to stay
under it. The run's peak is 87.71 GB allocated on cards of 95.6 GB, and section 11 measured the
worst case at 96,116 MiB of 97,887 live, i.e. 1.8 GB of headroom. That headroom is why this is
measured and not guessed: a threshold one bucket too high is an OOM forty minutes into an epoch.

WHAT IT DOES. Two short runs over the SAME units, differing only in --blocks-per-ckpt:

  A  --blocks-per-ckpt 1   (the live setting)
  B  --blocks-per-ckpt 0   (checkpointing off for everything)

--mem-log already prints `MEM <pdb>/<seed> pose <i> tokens <n>: allocated .. peak .. reserved ..`
after each pose's backward, so B's curve of peak-vs-tokens is read directly. The recommended
threshold is the largest token count whose peak in B stays under A's observed maximum minus
--margin-gb.

AND IT CHECKS THE EQUIVALENCE RATHER THAN ASSUMING IT. Checkpointing is gradient-exact and
test_perblock_ckpt_equiv.py already gates blocks_per_ckpt=1 against off at 0.000e+00 on every
parameter -- but that test runs three poses in isolation. This compares the two runs' per-pose
loss columns over every pose they both walked. A difference here means the recompute is NOT
reproducing the forward on this workload, and no timing number would be worth having.

Usage:
  python -I run_logging/bench_ckpt_by_size.py --units 24 --cards 4 --dry-run
  python -I run_logging/bench_ckpt_by_size.py --units 24 --cards 4
"""
import argparse
import csv
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_BIN = "/home/ebrahim/Protenix/.venv/bin/python"
# `pass` IS REQUIRED, NOT OPTIONAL. A pass A pose runs under no_grad and stores no activations;
# pooling its peak with a pass B pose's understates the one number the threshold is read off. An
# older trainer emitted no pass field, and this regex deliberately fails to match those lines
# rather than silently treating them as pass B -- a log with no match reports "no MEM lines
# parsed", which is true and loud, instead of a wrong peak.
MEM = re.compile(r"MEM (?P<pdb>\S+)/(?P<seed>\S+) pose (?P<pose>\d+) pass (?P<pass_>[AB]) "
                 r"tokens (?P<tok>-?\d+): "
                 r"allocated (?P<alloc>[\d.]+) GB\s+peak (?P<peak>[\d.]+)\s+"
                 r"reserved (?P<res>[\d.]+)")
EPOCH = re.compile(r"\[rank 0\] epoch \d+ took (?P<min>[\d.]+) min")
BUCKET = re.compile(r"(?P<k>gpu_backward|gpu_forward) (?P<m>[\d.]+)m/(?P<n>\d+)")

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", default="training/runs/jb5veo/resume_ep19.pt",
                help="the arm's bundle; every architecture setting is taken from its guard so "
                     "the benchmark cannot drift from the run it is for")
ap.add_argument("--units", type=int, default=24,
                help="structures to walk. They are the manifest's, so the token range is the "
                     "run's own, not a synthetic sweep.")
ap.add_argument("--cards", default="4",
                help="the card(s) made visible to the benchmark. ONE card is the default and the "
                     "right experiment: both numbers this measures -- per-pose peak memory and "
                     "per-pose device time in backward -- are rank-local, and the arm's own "
                     "gpu_backward/gpu_forward buckets are per-rank too, so a single rank is "
                     "directly comparable while adding no collective or barrier noise. Must name "
                     "cards nothing else holds; the standing limit is 3.")
ap.add_argument("--margin-gb", type=float, default=6.0,
                help="headroom demanded below the checkpointed peak. Section 11 measured 1.8 GB "
                     "of real headroom at the worst case, so the default is deliberately larger "
                     "than the slack: the threshold should not consume the margin that exists.")
ap.add_argument("--out", default="training/runs/ckpt_bench")
ap.add_argument("--master-port", type=int, default=29671,
                help="must not collide with the live arm's MASTER_PORT (29540 at the time of "
                     "writing; read it from /proc/<rank0>/environ before changing this)")
ap.add_argument("--probe", action="store_true",
                help="PROBE MODE instead of the three-arm comparison. Walks the manifest's "
                     "structures in ASCENDING token order, lean readouts on and checkpointing "
                     "OFF, and runs until it OOMs. The last pose that completed is then the "
                     "largest structure that survives uncheckpointed -- measured against the "
                     "real failure, an OOM, rather than against a peak-versus-budget proxy. One "
                     "process rather than one per probe: walking UP and letting it die costs a "
                     "single model load, where walking DOWN and restarting after each OOM costs "
                     "one per step.")
ap.add_argument("--probe-seeds", type=int, default=2,
                help="probe mode: seeds per structure. The arm uses 10, which would spend ~36 "
                     "min walking the 700-900 token region before reaching any boundary. 2 is "
                     "the MINIMUM that keeps the run intact: --rank-cross-seed forms its pairs "
                     "ACROSS seeds, so a single seed leaves it with no pair to rank and changes "
                     "the loss being measured. Peak memory is per-pose, so fewer seeds costs "
                     "nothing in what the probe can see.")
ap.add_argument("--start-tokens", type=int, default=700,
                help="probe mode: skip structures below this, so the walk starts near the "
                     "boundary instead of spending 20 minutes on units that obviously fit. The "
                     "manifest holds 136 structures at 700+ tokens and 6 at 1400+.")
ap.add_argument("--only", choices=("fat_ckpt_on", "lean_ckpt_on", "lean_ckpt_off",
                                  "fat_ckpt_off"),
                help="run ONE arm of the three-arm comparison and stop, instead of all three in "
                     "sequence. This is how the three arms are spread over three cards: launch "
                     "three invocations, each with its own --only and its own --cards, then run "
                     "a fourth with neither --only nor --cards changed -- run() reuses any "
                     "non-empty log it finds, so the fourth does no GPU work and only reports. "
                     "MASTER_PORT is offset per arm so concurrent invocations cannot collide.")
ap.add_argument("--set", action="append", default=[], metavar="FLAG=VALUE",
                help="override any flag in the arm's argv, repeatable. THE REASON THIS EXISTS: "
                     "the alignment-preflight marker is bound to a manifest digest and the "
                     "trainer refuses a mismatch, with no override -- by design, so a debug "
                     "population cannot authorise training. --limit-units and a reordered "
                     "manifest both change that digest, so the only way to shrink the work is to "
                     "shrink the POPULATION and re-run the preflight over it, then point the arm "
                     "at both: --set manifest-selection=<csv> --set alignment-preflight=<marker> "
                     "--set seeds=<range> --set limit-units=0. Applied after every other "
                     "override, so it wins.")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

_CARDS = [c for c in a.cards.split(",") if c.strip() != ""]
if not _CARDS:
    sys.exit("STOP: --cards is empty, so the benchmark would inherit the shell's view of the box "
             "and could run on every card on it.")
if len(_CARDS) > 3:
    sys.exit(f"STOP: --cards names {len(_CARDS)} card(s). The standing limit on this box is 3.")
if len(_CARDS) > 1:
    print(f"  NOTE: {len(_CARDS)} cards requested. WORLD_SIZE is set to {len(_CARDS)} but this "
          f"launches ONE process, so ranks 1..{len(_CARDS)-1} will never join and a collective "
          f"will hang. Pass a single card unless you also launch the other ranks.")

OUT = os.path.join(ROOT, a.out)


ARGV = os.path.join(ROOT, "training/runs/ckpt_bench/arm_argv.txt")


def bundle_epoch(path):
    """Epochs already completed in the bundle being resumed.

    WHY THIS IS NEEDED AT ALL. The epoch loop is `for ep in range(from_epoch, n_epochs)` with
    from_epoch taken from the bundle and n_epochs = ceil(--epochs). Passing --epochs 1 against a
    bundle at epoch 21 makes that range EMPTY: the run would exit having trained nothing, write
    no MEM line and no epoch-boundary timing line, and this benchmark would report "no MEM lines
    parsed" without ever saying why. So --epochs is set to the bundle's epoch + 1, which is one
    epoch of continuation. Found in review.
    """
    import torch
    b = torch.load(os.path.join(ROOT, path) if not os.path.isabs(path) else path,
                   map_location="cpu", weights_only=False)
    if not isinstance(b, dict) or "epoch" not in b:
        sys.exit(f"STOP: {path} carries no `epoch`, so the number of epochs to ask for cannot be "
                 f"derived from it. Keys: {sorted(b)[:12] if isinstance(b, dict) else type(b)}")
    return int(b["epoch"])


# READ EVEN ON A DRY RUN, so the printed command line is the one that would actually be sent and
# so an unreadable or wrong-shaped bundle is caught before any card is occupied.
BUNDLE_EP = bundle_epoch(a.bundle)
print(f"  bundle {a.bundle}: {BUNDLE_EP} epoch(s) done -> asking for --epochs {BUNDLE_EP + 1}")


def arm_argv():
    """The LIVE arm's own trainer argv, so the benchmark cannot drift from what it benchmarks.

    NOT RECONSTRUCTED FROM args.json. Every loss flag -- --rank-loss, --rank-form, --alpha-rank,
    --pae-only, the pose pathways -- has to be passed again on a resume, because the bundle's
    guard COMPARES them rather than supplying them, and an omitted one is a drift hard-stop, not
    a default. Rebuilding that line from parsed values means re-deriving every store_true and
    every default, and a single wrong one benchmarks a different model.
    """
    if not os.path.exists(ARGV):
        sys.exit(f"STOP: {os.path.relpath(ARGV, ROOT)} is missing. It is the arm's trainer argv, "
                 f"captured from /proc/<rank0>/cmdline while the run was up:\n"
                 f"  tr '\\0' '\\n' < /proc/<pid>/cmdline > {os.path.relpath(ARGV, ROOT)}")
    v = [x for x in open(ARGV).read().split("\n") if x != ""]
    if "--resume" not in v:
        sys.exit(f"STOP: {ARGV} has no --resume, so it is not a resume command line.")
    return v


def override(v, flag, value):
    """Replace a flag's value, or append the pair when the arm did not pass it."""
    if flag in v:
        i = v.index(flag)
        v[i + 1] = value
        return v
    return v + [flag, value]


# ONE PORT PER ARM. Each arm is WORLD_SIZE 1 and so forms no collective, but MASTER_PORT is
# still bound, and three concurrent --only invocations sharing one port would race on it. The
# offset is derived from the tag rather than passed in, so the caller cannot forget it.
_PORT_ORDER = ("fat_ckpt_on", "lean_ckpt_on", "lean_ckpt_off", "fat_ckpt_off", "probe")


def _port_offset(tag):
    # the probe tag carries its start token count ("probe_from700"), so match the prefix.
    if tag.startswith("probe"):
        return _PORT_ORDER.index("probe")
    return _PORT_ORDER.index(tag) if tag in _PORT_ORDER else 0


def run(tag, bpc, lean, extra=(), allow_oom=False):
    """One short run with --mem-log, the arm's argv otherwise.

    Three arms are compared, differing only in these two settings: see the ARMS table below.
    """
    work = os.path.join(a.out, tag)
    log = os.path.join(OUT, tag + ".log")
    # REUSE ONLY A LOG THAT ACTUALLY MEASURED SOMETHING. "Non-empty" is not the test: a run that
    # hard-stopped on a guard writes a long log with no MEM line in it, and reusing that is how a
    # crashed arm turns into a number in the report. The alignment-preflight stop found this.
    if os.path.exists(log) and os.path.getsize(log) > 0:
        if any(MEM.search(_l) for _l in open(log, errors="replace")):
            print(f"  {tag}: log exists with MEM lines, reusing")
            return log
        print(f"  {tag}: log exists but holds NO MEM line, so it measured nothing -- re-running. "
              f"Last line: {(open(log, errors='replace').read().strip().splitlines() or [''])[-1][:120]}")
    v = arm_argv()
    for flag, val in (("--resume", a.bundle), ("--blocks-per-ckpt", str(bpc)),
                      ("--limit-units", str(a.units)), ("--epochs", str(BUNDLE_EP + 1)),
                      ("--save-every", "0"), ("--out", work),
                      # NOT THE ARM'S EVAL. --train-eval-every would spend cards scoring 92
                      # held-out interfaces in the middle of a timing measurement.
                      ("--train-eval-every", "0")):
        v = override(v, flag, val)
    # A THRESHOLD IN THE ARM'S LINE WOULD MAKE BOTH ARMS OF THIS COMPARISON THE SAME RUN.
    if "--ckpt-token-threshold" in v:
        v = override(v, "--ckpt-token-threshold", "0")
    # --lean-readouts IS A store_true, so it is added or removed rather than given a value. The
    # arm's own line may or may not carry it; this makes the arm's state explicit either way.
    while "--lean-readouts" in v:
        v.remove("--lean-readouts")
    if lean:
        v.append("--lean-readouts")
    for _f, _val in extra:
        v = override(v, _f, _val)
    # LAST, so --set beats the arm's line and every override above.
    for _kv in a.set:
        if "=" not in _kv:
            sys.exit(f"STOP: --set {_kv!r} is not FLAG=VALUE.")
        _f, _val = _kv.split("=", 1)
        v = override(v, "--" + _f.lstrip("-"), _val)
    if "--mem-log" not in v:
        v.append("--mem-log")
    v = override(v, "--resume-note",
                 f"BENCHMARK ONLY (bench_ckpt_by_size.py), blocks_per_ckpt={bpc}, "
                 f"lean_readouts={lean}. This is a MEASUREMENT, not a continuation of the arm: "
                 f"it runs one epoch at WORLD_SIZE 1 instead of 3, over a truncated or "
                 f"size-ordered unit list, writes into {os.path.basename(a.out)}/ and never into "
                 f"the arm's run directory, and its optimiser trajectory is discarded -- no "
                 f"bundle it writes is ever resumed from and no scheduled step depends on it. "
                 f"The numbers taken from it are per-pose peak memory and CUDA-event device "
                 f"time, both of which are properties of one pose on one rank and so do not "
                 f"depend on the trajectory being scientifically valid.")
    # BLANKET DRIFT, DELIBERATELY, AND THIS IS THE ONE PLACE IT IS RIGHT.
    #
    # The guard compares world, seed, accum, nsample, manifest_digest and crop_size
    # (phase_k_train.py:2694) and refuses any that differ and are not named. This benchmark
    # changes world (3 -> 1) in every mode and, in probe mode, the manifest and the seed range
    # too -- so the exact field set depends on the mode, and a hardcoded --resume-override list
    # would either miss a field (a hard stop mid-benchmark) or name one that does not differ
    # (also a hard stop: a stale override is an error by design, phase_k_train.py:2910).
    # --resume-allow-drift plus the note above records the same thing without a list that goes
    # stale every time this script changes a flag.
    while "--resume-override" in v:
        _i = v.index("--resume-override")
        del v[_i:_i + 2]
    if "--resume-allow-drift" not in v:
        v.append("--resume-allow-drift")
    print(f"\n  === {tag} ===")
    # READ BACK FROM THE ARGV, not from a.units. extra= can override --limit-units after the
    # loop above has set it, and a print that reports the intent rather than the command is how
    # you end up certain you measured something you did not.
    print(f"    {len(v)} argv token(s) from the arm, " + "  ".join(
        f"{_f} {v[v.index(_f) + 1]}" for _f in
        ("--blocks-per-ckpt", "--limit-units", "--seeds", "--shuffle") if _f in v)
        + f"  lean_readouts={lean}")
    if a.dry_run:
        for _f in ("--lean-readouts", "--resume-allow-drift", "--mem-log", "--train-eval-every",
                   "--resume", "--epochs", "--out", "--resume-override"):
            if _f in v:
                _nx = v[v.index(_f) + 1] if v.index(_f) + 1 < len(v) else ""
                print(f"      {_f} {'' if _nx.startswith('--') else _nx}")
            else:
                print(f"      {_f} ABSENT")
        return None
    os.makedirs(OUT, exist_ok=True)
    # THE DEVICES AND THE RANK LAYOUT ARE SET HERE, NOT INHERITED. The arm's three ranks get
    # CUDA_VISIBLE_DEVICES/RANK/WORLD_SIZE from its launcher, never from the argv, so an argv
    # replayed out of a shell inherits NOTHING: the trainer would then see every card on the
    # box, which breaks the 3-card limit and can land on a card another user holds. A stale
    # WORLD_SIZE inherited from a parent is the same trap phase_k_train.py:1125 guards against.
    env = {**os.environ, "LAYERNORM_TYPE": "openfold",
           "CUDA_VISIBLE_DEVICES": a.cards,
           "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": str(len(a.cards.split(","))),
           "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(a.master_port + _port_offset(tag))}
    print(f"    CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
          f"WORLD_SIZE={env['WORLD_SIZE']} RANK=0")
    with open(log, "w") as fh:
        r = subprocess.run(v, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        # AN OOM IS THE PROBE'S ANSWER, NOT ITS FAILURE -- but ONLY an OOM. Any other nonzero
        # exit stays fatal: a Traceback, a guard hard-stop or a killed rank would otherwise be
        # read as "that is where memory ran out" and set a threshold from an unrelated crash.
        txt = open(log, errors="replace").read()
        oom = ("CUDA out of memory" in txt or "OutOfMemoryError" in txt
               or "CUDA error: out of memory" in txt)
        if not (allow_oom and oom):
            _why = "an OOM, which this mode expects" if oom else "NOT an OOM"
            sys.exit(f"STOP: {tag} exited {r.returncode} and it was {_why}; "
                     f"see {os.path.relpath(log, ROOT)}\n"
                     f"  last lines:\n    " + "\n    ".join(txt.strip().splitlines()[-6:]))
        print(f"    exited {r.returncode} on a CUDA OOM, which is the boundary being measured")
    return log


def parse(log, want_pass="B"):
    """MEM rows for ONE pass, plus the epoch buckets.

    want_pass="B" by default and that is the only value anything here asks for: the threshold is
    a statement about a pose that stores activations and backwards through them. Pass A poses run
    under no_grad, so their peak is roughly the resident model and inputs no matter how large the
    structure is, and pooling the two is how a 1162-token structure came to report the same peak
    as a 263-token one.
    """
    rows, buckets, took = [], {}, None
    n_other = 0
    for line in open(log, errors="replace"):
        m = MEM.search(line)
        if m:
            if m["pass_"] != want_pass:
                n_other += 1
                continue
            rows.append((int(m["tok"]), float(m["peak"]), float(m["alloc"]), float(m["res"]),
                         m["pdb"], m["seed"], int(m["pose"])))
        e = EPOCH.search(line)
        if e:
            took = float(e["min"])
            for b in BUCKET.finditer(line):
                buckets[b["k"]] = (float(b["m"]), int(b["n"]))
    print(f"    {os.path.basename(log)}: {len(rows)} pass-{want_pass} pose(s), "
          f"{n_other} other-pass pose(s) ignored")
    return rows, buckets, took


if a.probe:
    # A SIZE-ORDERED MANIFEST, built from the arm's own manifest so the structures, the clusters
    # and every other column are the run's. Only the ORDER and the subset change.
    src = None
    _v = arm_argv()
    if "--manifest-selection" in _v:
        src = _v[_v.index("--manifest-selection") + 1]
    if not src or not os.path.exists(os.path.join(ROOT, src)):
        sys.exit(f"STOP: probe mode needs the arm's --manifest-selection to exist; got {src!r}.")
    with open(os.path.join(ROOT, src)) as fh:
        rd = csv.DictReader(fh)
        hdr, rows = rd.fieldnames, list(rd)
    if "tokens" not in (hdr or []):
        sys.exit(f"STOP: {src} has no `tokens` column, so it cannot be ordered by size. "
                 f"Columns: {hdr}")
    keep = sorted((r for r in rows if int(r["tokens"]) >= a.start_tokens),
                  key=lambda r: int(r["tokens"]))
    if not keep:
        sys.exit(f"STOP: no structure in {src} has >= {a.start_tokens} tokens. The largest is "
                 f"{max(int(r['tokens']) for r in rows)}.")
    # TRIMMED FROM THE FRONT, SO THE LARGEST SURVIVE.
    #
    # build_epoch_cross_seed chunks the structures into steps of --rank-structs-per-step and
    # DROPS a short final chunk rather than padding it (phase_k_train.py: "A SHORT FINAL STEP IS
    # DROPPED"). On an ascending manifest that chunk is the LARGEST structures -- exactly the ones
    # a memory probe exists to reach -- so 136 structures at S=3 would silently never run the
    # biggest one. Removing the SMALLEST few instead makes the count a multiple of S and keeps
    # every large structure. Found in review.
    _S = 1
    if "--rank-structs-per-step" in _v:
        _S = max(1, int(_v[_v.index("--rank-structs-per-step") + 1]))
    _rem = len(keep) % _S
    if _rem:
        print(f"  trimming the {_rem} SMALLEST structure(s) "
              f"({[int(r['tokens']) for r in keep[:_rem]]} tokens) so {len(keep)} becomes "
              f"{len(keep) - _rem}, a multiple of --rank-structs-per-step {_S}: a short final "
              f"chunk is dropped by the epoch builder, and on an ascending manifest that would "
              f"be the LARGEST structures.")
        keep = keep[_rem:]
    if not keep:
        sys.exit(f"STOP: trimming to a multiple of {_S} left nothing. Lower --start-tokens.")
    os.makedirs(OUT, exist_ok=True)
    man = os.path.join(OUT, f"manifest_ascending_{a.start_tokens}.csv")
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(keep)
    print(f"  probe manifest: {len(keep)} structure(s), "
          f"{int(keep[0]['tokens'])} -> {int(keep[-1]['tokens'])} tokens -> "
          f"{os.path.relpath(man, ROOT)}")

    # --shuffle none KEEPS THE ORDER. With `structure` the trainer shuffles the structure order,
    # which would destroy the ascending walk the whole probe depends on.
    if a.probe_seeds < 2:
        sys.exit("STOP: --probe-seeds below 2 leaves --rank-cross-seed with no cross-seed pair, "
                 "so the probe would measure a different loss than the arm runs.")
    _v0 = arm_argv()
    _base = _v0[_v0.index("--seeds") + 1].split("-")[0] if "--seeds" in _v0 else "500000"
    extra = (("--manifest-selection", os.path.relpath(man, ROOT)),
             # --shuffle none IS NOT ENOUGH, which is the whole reason --epoch-order exists: the
             # shuffle inside build_epoch_cross_seed is unconditional and runs after it.
             ("--shuffle", "none"), ("--epoch-order", "manifest"),
             ("--seeds", f"{_base}-{int(_base) + a.probe_seeds - 1}"),
             ("--limit-units", str(len(keep) * a.probe_seeds + 1)))
    tag = f"probe_from{a.start_tokens}"
    log = run(tag, 0, True, extra=extra, allow_oom=True)
    if a.dry_run:
        print("\n  DRY RUN -- nothing executed")
        sys.exit(0)

    rows_, buckets_, took_ = parse(log)
    txt = open(log, errors="replace").read()
    oom = ("CUDA out of memory" in txt or "OutOfMemoryError" in txt)
    if not rows_:
        sys.exit(f"STOP: no MEM lines parsed from {os.path.relpath(log, ROOT)}. Either --mem-log "
                 f"did not fire or it died before the first pose.")

    # ---- THE REALISED ORDER, FROM THE TRAINER'S OWN OUTPUT -------------------------------
    #
    # NOT from the CSV. The CSV order was already correct when the builder's unconditional
    # shuffle was destroying it, so checking the CSV proves nothing. Under --epoch-order manifest
    # the chunks are consecutive and the units inside a chunk are appended per structure with no
    # shuffle, so the token sequence the trainer actually walked must be NON-DECREASING. If it is
    # not, the ordering did not take and no threshold may be read off this run.
    toks = [r[0] for r in rows_]
    bad = [(i, toks[i - 1], toks[i]) for i in range(1, len(toks)) if toks[i] < toks[i - 1]]
    if bad:
        sys.exit(
            f"STOP: the trainer did NOT walk the structures in ascending token order -- "
            f"{len(bad)} descent(s) in {len(toks)} pose(s), first at pose {bad[0][0]}: "
            f"{bad[0][1]} -> {bad[0][2]} tokens.\n"
            f"  --epoch-order manifest did not take, so the OOM below is not the boundary: it is "
            f"wherever the walk happened to reach a large structure. Do NOT set a threshold from "
            f"this run. Check that phase_k_train.py's builders honour a.epoch_order.")
    print(f"  ORDER VERIFIED from the trainer's own MEM lines: {len(toks)} pose(s), "
          f"{toks[0]} -> {toks[-1]} tokens, non-decreasing throughout")

    # ---- COVERAGE: which structures actually ran ----------------------------------------
    want = [(r["pdb"], int(r["tokens"])) for r in keep]
    got = {r[4] for r in rows_}
    missing = [(pdb, tk) for pdb, tk in want if pdb not in got]
    print(f"  coverage: {len(want) - len(missing)}/{len(want)} structure(s) produced a pose")
    if missing:
        # EXPECTED when the walk OOMs: everything above the boundary never ran. Printed as the
        # boundary's other side rather than as an error, and the SMALLEST missing structure is
        # the first one that did not complete.
        _sm = min(tk for _, tk in missing)
        print(f"  {len(missing)} structure(s) never produced a pose, the smallest at {_sm} "
              f"tokens" + ("  (expected: the walk OOMed below them)" if oom else
                           "  -- NOT expected without an OOM, so something else ended the walk"))
        if not oom:
            sys.exit("STOP: structures were skipped and the run did NOT OOM, so the walk ended "
                     "for a reason this script has not identified. No threshold from this run.")

    by_tok = {}
    for tok, peak, *_ in rows_:
        by_tok[tok] = max(by_tok.get(tok, 0.0), peak)
    seen = sorted(by_tok)
    print(f"\n  {'tokens':>8}{'peak GB':>10}")
    for tok in seen:
        print(f"  {tok:>8}{by_tok[tok]:>10.2f}")
    biggest = seen[-1]
    print(f"\n  largest structure that COMPLETED a pose uncheckpointed: {biggest} tokens "
          f"at {by_tok[biggest]:.2f} GB")
    if oom:
        print(f"  and it OOMed after that, which is the boundary this was looking for.")
    else:
        print(f"  NO OOM: the walk finished without one, so {biggest} tokens is the largest "
              f"the manifest HAS at or above --start-tokens, not the largest that fits. "
              f"The boundary is above the data -- checkpointing may be unnecessary outright.")
    # ONE BUCKET BELOW THE SURVIVOR, NOT AT IT. This process had a clean allocator; the live run
    # reaches that unit after thousands of steps of fragmentation, and the peak at the boundary is
    # not reproducible to the megabyte.
    edges = [0, 400, 500, 600, 700, 800, 1000, 1250, 1600]
    below = [e for e in edges if e < biggest]
    print(f"\n  RECOMMENDED --ckpt-token-threshold {below[-1] if below else 0}  "
          f"(one bucket below the largest survivor, for allocator fragmentation the probe's "
          f"fresh process does not have)")
    sys.exit(0)

# THE THREE ARMS.
#
#   fat_ckpt_on    what the live arm runs today. Its peak is the CEILING the other arms must stay
#                  under, because it is the only configuration demonstrated to survive an epoch.
#   lean_ckpt_on   --lean-readouts added, nothing else. The difference against fat_ckpt_on is the
#                  headroom the strip buys, which is the number the threshold gets to spend.
#   lean_ckpt_off  the configuration a small unit would actually run in. Its peak-vs-tokens curve
#                  is what the threshold is read off.
#
#   fat_ckpt_off   checkpointing off WITHOUT the strip. Added later: it is the control that
#                  isolates what the strip is worth TO THE THRESHOLD. lean_ckpt_off alone gives a
#                  boundary; the difference between the two boundaries is the number that decides
#                  whether --lean-readouts is what buys the threshold or whether turning
#                  checkpointing off would have bought it anyway. It is only reached with --only,
#                  so the default three-arm comparison is unchanged.
ARMS = (("fat_ckpt_on", 1, False), ("lean_ckpt_on", 1, True), ("lean_ckpt_off", 0, True),
        ("fat_ckpt_off", 0, False))
if a.only:
    # ONE ARM, THEN STOP. The comparison needs all three logs, so it is not attempted here; the
    # reporting invocation that follows finds this log already written and reuses it.
    _tag, _bpc, _lean = next(x for x in ARMS if x[0] == a.only)
    run(_tag, _bpc, _lean)
    if not a.dry_run:
        print(f"\n  {_tag} done -> {os.path.relpath(os.path.join(OUT, _tag + '.log'), ROOT)}")
        print(f"  --only was given, so the three-arm comparison was NOT run. Re-run without "
              f"--only once all three logs exist to get it.")
    sys.exit(0)
# THE REPORT IS STILL THE THREE-ARM ONE. fat_ckpt_off is an --only arm; including it here would
# make the default comparison spend a fourth run nothing below reads.
logs = {tag: run(tag, bpc, lean) for tag, bpc, lean in ARMS if tag != "fat_ckpt_off"}
if a.dry_run:
    print("\n  DRY RUN -- nothing executed")
    sys.exit(0)

F, bF, tF = parse(logs["fat_ckpt_on"])
A, bA, tA = parse(logs["lean_ckpt_on"])
B, bB, tB = parse(logs["lean_ckpt_off"])
for _t, _r in (("fat_ckpt_on", F), ("lean_ckpt_on", A), ("lean_ckpt_off", B)):
    if not _r:
        sys.exit(f"STOP: no MEM lines parsed for {_t}. --mem-log did not fire.")

peakF = max(r[1] for r in F)
peakA = max(r[1] for r in A)
print(f"\n  peak over {len(F)} pose(s), as the arm runs today: {peakF:.2f} GB")
print(f"  peak with --lean-readouts, checkpointing still on: {peakA:.2f} GB")
print(f"  HEADROOM THE STRIP BUYS: {peakF - peakA:+.2f} GB")
print(f"  budget for an uncheckpointed unit: {peakF - a.margin_gb:.2f} GB "
      f"(today's peak - {a.margin_gb:g} GB margin)\n")

# PEAK VS TOKENS, BINNED, so one outlier structure does not set the threshold on its own.
edges = [0, 400, 500, 600, 700, 800, 1000, 1250, 1600, 10 ** 9]
print(f"  {'tokens':<14}{'n':>5}{'peak OFF':>11}{'peak ON':>10}{'delta':>9}")
# AGAINST TODAY'S PEAK, not the lean one. The ceiling is the memory the run has been shown to
# survive; spending the strip's savings on a higher threshold is the entire point.
budget = peakF - a.margin_gb
ok_max = 0
for lo, hi in zip(edges, edges[1:]):
    gb = [r for r in B if lo <= r[0] < hi]
    ga = [r for r in A if lo <= r[0] < hi]
    if not gb:
        continue
    pb, pa = max(r[1] for r in gb), (max(r[1] for r in ga) if ga else float("nan"))
    fits = pb <= budget
    if fits:
        ok_max = max(ok_max, max(r[0] for r in gb))
    print(f"  {lo}-{hi if hi < 10**9 else '':<9}{len(gb):>5}{pb:>11.2f}{pa:>10.2f}"
          f"{pb - pa:>+9.2f}   {'fits' if fits else 'OVER BUDGET'}")

print(f"\n  RECOMMENDED --ckpt-token-threshold {ok_max}"
      if ok_max else "\n  RECOMMENDED: none -- no bucket fits the budget; leave checkpointing on")

if bA.get("gpu_backward") and bB.get("gpu_backward"):
    mA, nA = bA["gpu_backward"]
    mB, nB = bB["gpu_backward"]
    if nA != nB:
        print(f"\n  NOT COMPARABLE: {nA} backward(s) on, {nB} off -- the runs walked different "
              f"pose counts, so the totals are not a speedup.")
    else:
        print(f"\n  gpu_backward (CUDA events, {nA} pose(s)): on {mA:.1f}m  off {mB:.1f}m  "
              f"{(1 - mB / mA) * 100:+.1f}% when uncheckpointed")
        print(f"  gpu_forward: on {bA['gpu_forward'][0]:.1f}m  off {bB['gpu_forward'][0]:.1f}m")
        print(f"  NOTE: that is the saving on the units that QUALIFY, not on the epoch. Scale it "
              f"by the fraction of units at or below the threshold.")

# ---- THE EQUIVALENCE CHECK. Timing is worthless if the recompute is not reproducing the forward.
def losses(tag):
    p = os.path.join(ROOT, a.out, tag, "trajectory.csv")
    if not os.path.exists(p):
        return None
    out = {}
    for x in csv.DictReader(open(p)):
        try:
            out[(x["pdb"], x["unit"], x["pose"])] = float(x["loss"])
        except (ValueError, KeyError):
            pass
    return out


la, lb = losses("lean_ckpt_on"), losses("lean_ckpt_off")
if la and lb:
    keys = sorted(set(la) & set(lb))
    if not keys:
        print("\n  EQUIVALENCE: no shared poses -- NOT CHECKED")
    else:
        d = np.array([abs(la[k] - lb[k]) for k in keys])
        print(f"\n  EQUIVALENCE over {len(keys)} shared pose(s): max |loss_on - loss_off| "
              f"{d.max():.3e}, median {np.median(d):.3e}")
        if d.max() > 1e-4:
            sys.exit(f"STOP: checkpointing is NOT gradient-exact on this workload "
                     f"(max {d.max():.3e}). No timing number here is worth having.")
        print("  within tolerance: the recompute reproduces the forward.")
else:
    print("\n  EQUIVALENCE: a trajectory.csv is missing -- NOT CHECKED")
