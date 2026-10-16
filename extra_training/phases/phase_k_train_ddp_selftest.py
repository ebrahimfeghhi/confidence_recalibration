"""Does multi-GPU training compute the same thing as one GPU?

WHY THIS EXISTS. The multi-GPU path in phase_k_train.py is deliberately NOT
DistributedDataParallel: the forward is RC.confidence(model, configs, ...) over a frozen trunk, and
DDP's hooks would both fight that custom call and expect gradients for the 444M parameters that have
none. Instead every rank holds an identical replica and the HEAD gradients are summed across ranks
before the optimiser step. Three things make that equal to one GPU, and each fails silently:

  1. THE LOSS SCALE IS THE GLOBAL ACCUMULATION. Each pose is divided by accum_eff, not by
     accum_local, so summing the ranks reconstructs the full-batch mean. Dividing by accum_local
     instead makes the gradient `world_size` too large.
  2. THE ALL-REDUCE HAPPENS BEFORE THE CLIP. Clipping each rank's partial gradient to norm 1 and
     then summing is a different function from clipping the sum.
  3. EVERY RANK TAKES THE SAME NUMBER OF STEPS, or the first mismatched collective hangs forever.

WHY AN EARLIER VERSION OF THIS TEST WAS NOT EVIDENCE. It compared the two runs' final parameters
against a tolerance relative to the PARAMETER magnitude: 1e-4 x 1.04e+02 = 1.04e-02 absolute. One
Adam step at lr 1e-3 moves a parameter by about 1e-3. So a multi-GPU run that took NO STEP AT ALL
differed from the single-GPU run by ~1e-3 and PASSED, by a factor of ten. Worse for requirement 1:
Adam's update is m_hat/sqrt(v_hat), which is invariant to a constant factor on the gradient, so after
one step a gradient scaled by `world_size` produces very nearly the same parameters. The single thing
the test most needed to catch was the thing it was least able to see.

WHAT THIS VERSION CHECKS, and why each check can actually fail:

  SCALE, on the gradient itself. clip_grad_norm_ returns the total norm BEFORE clipping, which the
  trainer records per step as `grad_norm`. That number is linear in the gradient, so it is immune to
  Adam's normalisation: dividing by accum_local instead of accum_eff shows up as a ratio of exactly
  `world_size`. This is the check that covers requirement 1, and nothing about the parameters can
  substitute for it.

  THE UPDATE, not the parameters. The comparison is between the two runs' DISPLACEMENTS from the
  initial head, and the tolerance is relative to the size of that displacement. A run that did
  nothing has zero displacement and fails by a relative error of 1.0.

  SENSITIVITY, asserted rather than assumed. The test computes what it WOULD have reported for a
  multi-GPU run that took no step, and fails if that case would have passed. A test whose negative
  control passes is not a test.

  THE REPLICAS STAY TOGETHER over several steps. Exact agreement with one GPU is only meaningful for
  a single full-batch step: with more steps the two runs partition the poses differently -- one GPU
  takes consecutive poses, the ranks take one from each structure -- so their trajectories diverge
  legitimately. What must hold over many steps is that the RANKS agree with EACH OTHER, since they
  are supposed to be identical replicas. A missed all-reduce would show up here and nowhere else.

THE RUNS. Two structures with one seed each are 2 units of 5 poses = 10 poses.

    INIT   1 GPU,  --lr 0   -> the initial head, so displacements can be measured
    ONE    1 GPU,  --accum 10 -> ONE step over all 10 poses
    TWO    2 GPUs, --accum 10 -> accum_local 5, accum_eff 10, ONE step over the same 10
    MANY   2 GPUs, --accum 2  -> 5 steps, every rank writing its own head

It needs two IDLE cards and defaults to 1,3 so it can run beside an evaluator on GPU 0.
"""
import argparse, csv, os, shutil, statistics, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/ebrahim/Protenix/.venv/bin/python"
TORCHRUN = "/home/ebrahim/Protenix/.venv/bin/torchrun"


def selection_file(work, pdbs):
    """A two-structure manifest, from the frozen selection rather than typed out here."""
    src = os.path.join(ROOT, "train_val_splits", "cache_train_structures.csv")
    rows = open(src).read().splitlines()
    keep = [rows[0]] + [r for r in rows[1:] if len(r.split(",")) > 1 and r.split(",")[1] in pdbs]
    if len(keep) != len(pdbs) + 1:
        sys.exit(f"STOP: wanted {sorted(pdbs)} in {src}, found {len(keep) - 1}")
    out = os.path.join(work, "sel.csv")
    open(out, "w").write("\n".join(keep) + "\n")
    return out


def missing_inputs(pdbs, seed):
    out = []
    for p in pdbs:
        for kind in ("labels", "trunks"):
            f = os.path.join(ROOT, "training", "fold", kind, p, f"free_{seed}.npz")
            if not os.path.exists(f):
                out.append(f)
    return out


def grad_norms(run_dir):
    """The PRE-CLIP gradient norm of each optimiser step, from the trajectory.

    clip_grad_norm_ returns the norm before it clips, and the trainer records it on the pose that
    triggered the step, so `step_taken == 1` selects exactly the steps.
    """
    p = os.path.join(run_dir, "trajectory.csv")
    if not os.path.exists(p):
        return []
    out = []
    for r in csv.DictReader(open(p)):
        if r.get("step_taken") == "1" and r.get("nonfinite_grad") != "1":
            try:
                out.append(float(r["grad_norm"]))
            except (KeyError, ValueError):
                pass
    return out


def load_head(path):
    import torch
    return torch.load(path, map_location="cpu")


def delta(a, b):
    """a - b, as a flat float64 vector over the tensors they share, in a fixed order."""
    import torch
    keys = sorted(set(a) & set(b))
    return torch.cat([(a[k].double() - b[k].double()).reshape(-1) for k in keys])


def run(tag, cmd, env, log, timeout):
    print(f"== {tag} ==", flush=True)
    with open(log, "w") as fh:
        try:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(open(log).read()[-1200:])
            sys.exit(f"FAIL: {tag} did not finish within {timeout}s. Requirement 3 -- a rank-count "
                     f"mismatch deadlocks on the collective rather than raising -- is the first "
                     f"thing to check: the shard sizes and the per-unit pose count.")
    print(f"   exit={r.returncode}")
    if r.returncode:
        print(open(log).read()[-1500:])
        sys.exit(f"FAIL: {tag} did not finish")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="1,3", help="two IDLE cards; GPU 0 is left alone by default")
    ap.add_argument("--pdbs", default="9g13,8h3c")
    ap.add_argument("--seed", default="500000")
    ap.add_argument("--port", default="29551")
    ap.add_argument("--lr", default="1e-3")
    ap.add_argument("--tol-update", type=float, default=1e-3,
                    help="RELATIVE to the size of the UPDATE, not to the parameter. float32 "
                         "summation order lands near 1e-5 of the update; a no-op scores 1.0.")
    ap.add_argument("--tol-grad", type=float, default=1e-4,
                    help="relative tolerance on the PRE-CLIP gradient norm. A loss scaled by "
                         "accum_local instead of accum_eff scores world_size - 1, i.e. 1.0 on two "
                         "ranks.")
    ap.add_argument("--tol-rank", type=float, default=1e-9,
                    help="relative tolerance BETWEEN RANKS. They run identical arithmetic on "
                         "identical gradients, so this is near-exact by construction, not by luck.")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    pdbs = [x.strip() for x in a.pdbs.split(",") if x.strip()]
    cards = [x.strip() for x in a.cards.split(",") if x.strip()]
    if len(pdbs) != 2 or len(cards) != 2:
        sys.exit("STOP: this test is built around exactly two structures on two cards")
    miss = missing_inputs(pdbs, a.seed)
    if miss:
        sys.exit(f"STOP: {len(miss)} input(s) absent, e.g. {miss[:2]}. Pick other --pdbs.")

    work = tempfile.mkdtemp(prefix="ddp_equiv_")
    dirs = {k: os.path.join(work, k) for k in ("init", "one", "two", "many")}
    for d in dirs.values():
        os.makedirs(d)
    sel = selection_file(work, set(pdbs))
    train = os.path.join(ROOT, "phases", "phase_k_train.py")
    base = ["--fold", f"{ROOT}/training/fold", "--labels", f"{ROOT}/training/fold/labels",
            "--manifest-selection", sel, "--seeds", f"{a.seed}-{a.seed}",
            "--manifest-skip", "9mx1", "--epochs", "1", "--warmup-steps", "1",
            "--preload-items", "2", "--grad-check-every", "1", "--save-every", "0",
            "--require-replay", "skip", "--allow-code-drift"]
    env = {**os.environ, "LAYERNORM_TYPE": "openfold"}
    e1 = {**env, "CUDA_VISIBLE_DEVICES": cards[0]}
    e2 = {**env, "CUDA_VISIBLE_DEVICES": a.cards}
    tr2 = [TORCHRUN, "--nproc_per_node=2", f"--master_port={a.port}", train]

    failures = []
    try:
        run("INIT  (1 GPU, lr 0 -- the starting head)",
            [PY, "-I", train, *base, "--accum", "10", "--lr", "0", "--out", dirs["init"]],
            e1, f"{dirs['init']}/run.log", a.timeout)
        run(f"ONE   (1 GPU, accum 10, lr {a.lr})",
            [PY, "-I", train, *base, "--accum", "10", "--lr", a.lr, "--out", dirs["one"]],
            e1, f"{dirs['one']}/run.log", a.timeout)
        run(f"TWO   (2 GPUs, accum 10, lr {a.lr})",
            [*tr2, *base, "--accum", "10", "--lr", a.lr, "--out", dirs["two"]],
            e2, f"{dirs['two']}/run.log", a.timeout)
        run(f"MANY  (2 GPUs, accum 2 -> 5 steps, lr {a.lr})",
            [*tr2, *base, "--accum", "2", "--lr", a.lr, "--debug-save-rank-heads",
             "--out", dirs["many"]], e2, f"{dirs['many']}/run.log", a.timeout)

        H = {k: load_head(os.path.join(dirs[k], "head_final.pt")) for k in ("init", "one", "two")}

        # ---- 0. the lr-0 run really is the starting head, or every displacement below is wrong ----
        d_init = delta(H["init"], H["one"]).abs().max().item()
        print("\n== 0. the lr-0 run is a fixed point ==")
        print(f"   |INIT - ONE| max = {d_init:.3e}  (must be > 0: otherwise lr {a.lr} did nothing)")
        if d_init == 0.0:
            failures.append("the lr-0 run and the trained run are identical, so either the training "
                            "run took no step or --lr 0 is not a fixed point; nothing below means "
                            "anything")

        # ---- 1. SCALE: the pre-clip gradient norm, which Adam cannot hide ----
        g1, g2 = grad_norms(dirs["one"]), grad_norms(dirs["two"])
        print("\n== 1. SCALE: pre-clip gradient norm (requirement 1) ==")
        print(f"   1 GPU : {['%.6f' % x for x in g1]}")
        print(f"   2 GPUs: {['%.6f' % x for x in g2]}")
        if len(g1) != 1 or len(g2) != 1:
            failures.append(f"expected exactly one optimiser step in each full-batch run, got "
                            f"{len(g1)} and {len(g2)}; the step accounting is not what this test "
                            f"assumes")
        else:
            rel_g = abs(g1[0] - g2[0]) / max(abs(g1[0]), 1e-30)
            print(f"   relative difference: {rel_g:.3e}  (tolerance {a.tol_grad:.0e})")
            print(f"   a loss scaled by accum_local instead of accum_eff would score 1.0e+00 here")
            if rel_g > a.tol_grad:
                failures.append(f"pre-clip gradient norms differ by {rel_g:.3e}, over "
                                f"{a.tol_grad:.0e}: the multi-GPU gradient is not the single-GPU "
                                f"gradient. A ratio near {len(g2) and 2} means the loss scale is "
                                f"per-rank rather than global")

        # ---- 2. THE UPDATE, with its own sensitivity control ----
        dv1 = delta(H["one"], H["init"])
        dv2 = delta(H["two"], H["init"])
        n1 = dv1.norm().item()
        n2 = dv2.norm().item()
        rel_u = (dv1 - dv2).norm().item() / max(n1, 1e-30)
        noop = dv1.norm().item() / max(n1, 1e-30)          # what a do-nothing run would score: 1.0
        print("\n== 2. THE UPDATE: displacement from the starting head ==")
        print(f"   ||ONE  - INIT|| = {n1:.6e}")
        print(f"   ||TWO  - INIT|| = {n2:.6e}")
        print(f"   relative difference of the updates: {rel_u:.3e}  (tolerance {a.tol_update:.0e})")
        print(f"   a multi-GPU run that took NO step would score {noop:.3e}")
        if rel_u > a.tol_update:
            failures.append(f"the updates differ by {rel_u:.3e} relative, over {a.tol_update:.0e}")
        if not noop > a.tol_update:
            failures.append("the negative control PASSES: a multi-GPU run that took no step at all "
                            "would be accepted, so this check proves nothing")

        # ---- 3. THE REPLICAS, over several steps ----
        rank_files = sorted(f for f in os.listdir(dirs["many"])
                            if f.startswith("head_final_rank") and f.endswith(".pt"))
        steps_many = len(grad_norms(dirs["many"]))
        print(f"\n== 3. THE REPLICAS after {steps_many} step(s) ==")
        print(f"   per-rank heads: {rank_files}")
        if len(rank_files) < 2:
            failures.append(f"expected a head per rank from the multi-step run, got {rank_files}")
        elif steps_many < 2:
            failures.append(f"the multi-step run took {steps_many} step(s); it is meant to take "
                            f"several, so replica drift has no chance to appear")
        else:
            heads = [load_head(os.path.join(dirs["many"], f)) for f in rank_files]
            ref = delta(heads[0], H["init"]).norm().item()
            worst = 0.0
            for h in heads[1:]:
                worst = max(worst, delta(heads[0], h).norm().item())
            rel_r = worst / max(ref, 1e-30)
            print(f"   ||rank0 - rankN|| = {worst:.3e},  relative to the update {rel_r:.3e}  "
                  f"(tolerance {a.tol_rank:.0e})")
            if rel_r > a.tol_rank:
                failures.append(f"the ranks disagree by {rel_r:.3e} relative after {steps_many} "
                                f"steps: the replicas are drifting, which means a step is being "
                                f"taken on an un-reduced gradient")

        print()
        if failures:
            for f in failures:
                print(f"FAIL: {f}")
            sys.exit(1)
        print(f"PASS: the multi-GPU gradient matches the single-GPU gradient to {a.tol_grad:.0e}, "
              f"the update to {a.tol_update:.0e}, the replicas agree after {steps_many} steps, and "
              f"a do-nothing run would have been rejected")
    finally:
        if a.keep:
            print(f"scratch kept at {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
