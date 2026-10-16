"""Does multi-GPU training compute the same thing as one GPU?

WHY THIS EXISTS. The multi-GPU path in phase_k_train.py is deliberately NOT
DistributedDataParallel: the forward is RC.confidence(model, configs, ...) over a frozen trunk, and
DDP's hooks would both fight that custom call and expect gradients for the 444M parameters that
have none. Instead every rank holds an identical replica and the HEAD gradients are summed across
ranks before the optimiser step. That is equivalent to a single GPU only if three things hold, and
each of them is easy to get wrong in a way no loss curve would reveal:

  1. THE LOSS SCALE IS THE GLOBAL ACCUMULATION. Each pose is divided by the effective batch, not by
     the per-rank batch, so summing the ranks reconstructs the full-batch mean. Divide by the
     per-rank batch instead and the gradient is `world_size` times too large -- which looks like a
     learning rate that is simply too high, and would be "fixed" by lowering the learning rate.
  2. THE ALL-REDUCE HAPPENS BEFORE THE CLIP. Clipping each rank's partial gradient to norm 1 and
     then summing is a different function from clipping the sum, so the multi-GPU run would quietly
     optimise something else.
  3. EVERY RANK TAKES THE SAME NUMBER OF STEPS. If they do not, the first mismatched collective
     hangs forever instead of failing.

HOW IT TESTS THEM. Two structures with one seed each are 2 units of 5 poses = 10 poses:

    one GPU  : --accum 10              -> ONE step over all 10 poses
    two GPUs : one structure per rank   -> accum_local 5, accum_eff 10, ONE step over the same 10

Same poses, same single step, so the two runs must end with the same head parameters. Points 1 and
2 are what that comparison is sensitive to: a per-rank loss scale shows up as a gradient scaled by
the world size, and clipping on the wrong side of the collective changes the step whenever the
gradient norm exceeds 1. Point 3 shows up as the two-GPU run never finishing, which the timeout
catches.

WHAT "THE SAME" MEANS. Not bit-identical. float32 addition is not associative and the two runs sum
the poses in a different order, so the measured difference is ~8e-06 on parameters whose mean
magnitude is ~1.0e+02, about 1e-07 relative. The tolerance is 1e-4 relative: far above summation
noise, far below anything a real gradient error could hide under.

It also asserts that exactly ONE rank writes head_final.pt. Both did before the rank-0 guard; the
content was identical because the replicas agree, but two processes writing one path can interleave
and leave a truncated checkpoint.

It needs two IDLE cards and defaults to 1,3 so it can run beside an evaluator on GPU 0.
"""
import argparse, os, shutil, statistics, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/ebrahim/Protenix/.venv/bin/python"
TORCHRUN = "/home/ebrahim/Protenix/.venv/bin/torchrun"


def selection_file(work, pdbs):
    """A two-structure manifest, from the frozen selection rather than typed out here."""
    src = os.path.join(ROOT, "train_val_splits", "cache_train_structures.csv")
    rows = open(src).read().splitlines()
    out = os.path.join(work, "sel.csv")
    keep = [rows[0]] + [r for r in rows[1:] if len(r.split(",")) > 1 and r.split(",")[1] in pdbs]
    if len(keep) != len(pdbs) + 1:
        sys.exit(f"STOP: wanted {pdbs} in {src}, found {len(keep) - 1}")
    open(out, "w").write("\n".join(keep) + "\n")
    return out


def have_inputs(pdbs, seed):
    """Both structures need a label AND a trunk for the seed, or the test is measuring nothing."""
    missing = []
    for p in pdbs:
        for kind in ("labels", "trunks"):
            f = os.path.join(ROOT, "training", "fold", kind, p, f"free_{seed}.npz")
            if not os.path.exists(f):
                missing.append(f)
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="1,3", help="two IDLE cards; GPU 0 is left alone by default")
    ap.add_argument("--pdbs", default="9g13,8h3c")
    ap.add_argument("--seed", default="500000")
    ap.add_argument("--port", default="29551")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="RELATIVE tolerance. float32 summation order lands near 1e-7.")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="a rank-count mismatch deadlocks rather than failing, so the two-GPU run "
                         "is bounded in time and a timeout is reported as that.")
    ap.add_argument("--keep", action="store_true", help="keep the scratch runs for inspection")
    a = ap.parse_args()

    import torch                                    # imported late so --help needs no GPU
    pdbs = [x.strip() for x in a.pdbs.split(",") if x.strip()]
    cards = [x.strip() for x in a.cards.split(",") if x.strip()]
    if len(pdbs) != 2 or len(cards) != 2:
        sys.exit("STOP: this test is built around exactly two structures on two cards")
    miss = have_inputs(pdbs, a.seed)
    if miss:
        sys.exit(f"STOP: {len(miss)} input(s) absent, e.g. {miss[:2]}. Pick other --pdbs.")

    work = tempfile.mkdtemp(prefix="ddp_equiv_")
    one, two = os.path.join(work, "one_gpu"), os.path.join(work, "two_gpu")
    os.makedirs(one); os.makedirs(two)
    sel = selection_file(work, set(pdbs))

    common = ["--fold", f"{ROOT}/training/fold", "--labels", f"{ROOT}/training/fold/labels",
              "--manifest-selection", sel, "--seeds", f"{a.seed}-{a.seed}",
              "--manifest-skip", "9mx1", "--epochs", "1", "--accum", "10", "--lr", "1e-3",
              "--warmup-steps", "1", "--preload-items", "2", "--grad-check-every", "1",
              "--save-every", "0", "--require-replay", "skip", "--allow-code-drift"]
    train = os.path.join(ROOT, "phases", "phase_k_train.py")
    rc = 1
    try:
        env = {**os.environ, "LAYERNORM_TYPE": "openfold"}
        print(f"== one GPU (card {cards[0]}) ==", flush=True)
        with open(f"{one}/run.log", "w") as lg:
            r = subprocess.run([PY, "-I", train, *common, "--out", one], stdout=lg,
                               stderr=subprocess.STDOUT,
                               env={**env, "CUDA_VISIBLE_DEVICES": cards[0]}, timeout=a.timeout)
        print(f"   exit={r.returncode}")
        if r.returncode:
            print(open(f"{one}/run.log").read()[-800:])
            sys.exit("FAIL: the single-GPU reference did not finish")

        print(f"== two GPUs (cards {a.cards}) ==", flush=True)
        with open(f"{two}/run.log", "w") as lg:
            try:
                r = subprocess.run([TORCHRUN, "--nproc_per_node=2", f"--master_port={a.port}",
                                    train, *common, "--out", two], stdout=lg,
                                   stderr=subprocess.STDOUT,
                                   env={**env, "CUDA_VISIBLE_DEVICES": a.cards}, timeout=a.timeout)
            except subprocess.TimeoutExpired:
                sys.exit(f"FAIL: the two-GPU run did not finish within {a.timeout}s. A rank-count "
                         f"mismatch deadlocks on the collective rather than raising; check the "
                         f"shard sizes and the per-unit pose count.")
        print(f"   exit={r.returncode}")
        if r.returncode:
            print(open(f"{two}/run.log").read()[-1500:])
            sys.exit("FAIL: the two-GPU run did not finish")

        saves = sum(1 for ln in open(f"{two}/run.log")
                    if "saved" in ln and "head_final" in ln)
        print(f"== writers of head_final.pt: {saves} (must be 1) ==")
        if saves != 1:
            sys.exit(f"FAIL: {saves} rank(s) wrote the final checkpoint; concurrent writes to one "
                     f"path can leave it truncated")

        A = torch.load(f"{one}/head_final.pt", map_location="cpu")
        B = torch.load(f"{two}/head_final.pt", map_location="cpu")
        if set(A) != set(B):
            sys.exit(f"FAIL: different tensor sets ({len(A)} vs {len(B)})")
        worst, where = 0.0, ""
        for k in sorted(A):
            d = (A[k].float() - B[k].float()).abs().max().item()
            if d > worst:
                worst, where = d, k
        mag = statistics.mean(A[k].float().abs().mean().item() for k in A)
        rel = worst / mag if mag else float("inf")
        print(f"== {len(A)} tensors compared ==")
        print(f"   worst absolute difference : {worst:.3e}  ({where})")
        print(f"   mean |parameter|          : {mag:.3e}")
        print(f"   relative                  : {rel:.2e}  (tolerance {a.tol:.0e})")
        if rel > a.tol:
            sys.exit(f"FAIL: relative difference {rel:.2e} exceeds {a.tol:.0e} -- the multi-GPU "
                     f"gradient is not the single-GPU gradient")
        print("PASS: multi-GPU training reproduces the single-GPU head to float32 summation order")
        rc = 0
    finally:
        if a.keep:
            print(f"scratch kept at {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
