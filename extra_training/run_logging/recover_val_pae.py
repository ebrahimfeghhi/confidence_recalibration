#!/usr/bin/env python
"""Re-run the held-out PAE shards that FAILED for one epoch, then pool the epoch's number.

WHY THIS EXISTS. phase_k_train voids an epoch's held-out point if ANY shard fails, which is the
right call -- pooling the survivors would report a mean over a biased subset of the held-out set,
a different population than every other epoch, and it would look entirely comparable. But the
surviving shards' work is still on disk, and the bundle the failed ones were measuring
(resume_ep<N>.pt) is too. So the point is recoverable by re-running ONLY the failed shards against
the SAME bundle and pooling all three, which is what this does.

tv6hue epoch 4 lost shards 1 and 2 to CUDA driver init failures ("CUDA driver initialization
failed" and "No CUDA GPUs are available") while shard 0 completed normally in 2.1 min at 3.80 GB.

IT REBUILDS THE PARENT'S OWN COMMAND from the run's args.json, mirroring heldout_pae()'s
construction flag for flag. It does NOT invent a command: any flag the parent passes that this
misses would measure a different model or a different population, and the number would look fine.
That is why --verify-against exists and why it is on by default -- after a shard succeeds, its
written args.json is compared field by field against a shard that the PARENT launched, and the
only fields allowed to differ are the ones that identify the shard.

POOLING REPEATS THE PARENT'S GUARDS rather than trusting the re-run: disjointness across shards,
one coverage line per shard, agreement on the full population, every unit accounted for, and an
exact pose count. A recovered number that cannot pass the checks the live path passes is not a
recovered number.
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "phases", "phase_k_train.py")

# Fields that legitimately differ between two shards of the same pass.
SHARD_FIELDS = {"out", "manifest_shard", "resume_note"}


def build_cmd(a, bundle, out, epoch):
    """Mirror phase_k_train.heldout_pae()'s cmd, which is the only correct source for this."""
    labels = a.get("val_loss_labels") or os.path.join(a["val_loss_fold"], "labels")
    cmd = [sys.executable, "-I", TRAIN,
           "--loss-only", "--resume", bundle, "--resume-allow-drift",
           "--resume-note", (f"held-out PAE measurement of {os.path.basename(bundle)} at "
                             f"epoch {epoch}; --loss-only reads the weights and writes no "
                             f"checkpoint, so the bundle's schedule is not continued"),
           "--fold", a["val_loss_fold"], "--labels", labels,
           "--manifest-selection", a["val_loss_selection"],
           "--alignment-preflight", a["val_loss_preflight"], "--minimal-items",
           "--pae-scored-pairs", a["val_loss_pool"],
           "--seeds", a["val_loss_seeds"],
           "--crop-margin", ("0" if a.get("val_loss_uncropped") else str(a["crop_margin"])),
           "--crop-contact", str(a["crop_contact"]),
           "--crop-size", str(a["crop_size"]), "--accum", str(a["accum"])]
    if a.get("val_loss_emit_iptm"):
        # PER SHARD. This had the shared val_loss_ep<N> directory, so several recovered shards
        # would open ONE file with "w" and truncate each other -- the exact bug 511519e7 fixed in
        # the trainer, reintroduced here. Caught in review.
        cmd += ["--emit-iptm-rows", os.path.join(out, "iptm_rows.csv")]
    if a.get("crop_all_poses") and not a.get("val_loss_uncropped"):
        cmd.append("--crop-all-poses")
    cmd += ["--nsample", str(a["nsample"]),
            "--alpha-pae", str(a["alpha_pae"] if float(a["alpha_pae"]) > 0 else 1.0),
            "--alpha-confidence", str(a["alpha_confidence"]), "--clip-norm", str(a["clip_norm"]),
            "--blocks-per-ckpt", str(a["blocks_per_ckpt"]),
            "--epochs", "1", "--save-every", "0", "--warmup-steps", "0",
            "--shuffle", "structure", "--out", out]
    for flag, key in (("--pae-only", "pae_only"), ("--drop-pde", "drop_pde"),
                      ("--frame-direction", "frame_direction")):
        if a.get(key):
            cmd.append(flag)
    if a.get("pose_perblock"):
        cmd.append("--pose-perblock")
        for flag, key in (("--pose-perblock-tied", "pose_perblock_tied"),
                          ("--pose-perblock-share", "pose_perblock_share"),
                          ("--pose-perblock-copy", "pose_perblock_copy")):
            if a.get(key):
                cmd.append(flag)
    if a.get("val_loss_prep"):
        p = a["val_loss_prep"]
        cmd += ["--indices", os.path.join(p, "job_indices.csv"),
                "--bio-dir", os.path.join(p, "job_bioassembly"),
                "--mmcif-dir", os.path.join(p, "job_cifs")]
    return cmd


def child_env(card):
    env = {k: v for k, v in os.environ.items()
           if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK", "ROLE_RANK",
                        "ROLE_NAME", "LOCAL_WORLD_SIZE", "ROLE_WORLD_SIZE", "MASTER_ADDR",
                        "MASTER_PORT", "TORCH_NCCL_ASYNC_ERROR_HANDLING")
           and not k.startswith("TORCHELASTIC_")}
    env.update({"LAYERNORM_TYPE": "openfold", "WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0",
                "CUDA_VISIBLE_DEVICES": str(card)})
    return env


def pool(shard_dirs, nsh):
    """The parent's pooling and ALL of its guards. Returns (mean, npose) or raises."""
    vals, ids, cov = [], set(), {}
    for i, d in sorted(shard_dirs.items()):
        lg = os.path.join(d, "shard.log")
        with open(lg) as fh:
            for ln in fh:
                if ln.startswith("SHARD COVERAGE "):
                    cov[i] = dict(kv.split("=", 1) for kv in ln.split()[2:] if "=" in kv)
        with open(os.path.join(d, "trajectory.csv")) as fh:
            for row in csv.DictReader(fh):
                if not row.get("pae_loss"):
                    continue
                k = (row.get("pdb"), row.get("unit"), row.get("pose"))
                if k in ids:
                    raise SystemExit(f"STOP: pose {k} appears in more than one shard; "
                                     f"the shards are not disjoint.")
                ids.add(k)
                vals.append(float(row["pae_loss"]))
    if len(cov) != nsh:
        raise SystemExit(f"STOP: coverage lines from {len(cov)} of {nsh} shard(s).")
    alls = {int(c["units_all"]) for c in cov.values()}
    nsamp = {int(c["nsample"]) for c in cov.values()}
    if len(alls) != 1 or len(nsamp) != 1:
        raise SystemExit(f"STOP: shards disagree on the population (units_all={alls}, "
                         f"nsample={nsamp}).")
    exp_units, ns = alls.pop(), nsamp.pop()
    mine = sum(int(c["units_mine"]) for c in cov.values())
    if mine != exp_units:
        raise SystemExit(f"STOP: shards cover {mine} of {exp_units} unit(s).")
    exp = exp_units * ns
    if len(vals) != exp or len(ids) != exp:
        raise SystemExit(f"STOP: pooled {len(vals)} pose(s) ({len(ids)} unique) against {exp} "
                         f"expected ({exp_units} x {ns}).")
    return sum(vals) / len(vals), len(vals)


def bundle_step(bundle, run, epoch):
    """The optimiser step this epoch ended on, from the bundle itself where possible.

    The trainer writes `step` into val_loss.csv from its live counter. Recovering after the fact
    has to get it from somewhere else, and the honest source is the checkpoint: it carries the
    step it was saved at. A head_ep<N>_step<S>.pt beside it is used as a cross-check, and a
    disagreement raises rather than picking one -- a wrong step column would silently misplace
    the point on every step-indexed plot.
    """
    from_ckpt = None
    try:
        import torch
        b = torch.load(bundle, map_location="cpu", weights_only=False)
        if isinstance(b, dict):
            for k in ("step", "global_step", "optimizer_step"):
                if isinstance(b.get(k), int):
                    from_ckpt = b[k]
                    break
    except Exception as e:                        # noqa: BLE001
        print(f"  (could not read a step from the bundle: {type(e).__name__}: {e})")
    from_name = None
    for f in os.listdir(run):
        m = re.fullmatch(rf"head_ep{epoch}_step(\d+)\.pt", f)
        if m:
            from_name = int(m.group(1))
            break
    if from_ckpt is not None and from_name is not None and from_ckpt != from_name:
        raise SystemExit(f"STOP: bundle says step {from_ckpt} but head_ep{epoch}_step{from_name}.pt "
                         f"says {from_name}; refusing to guess the step column.")
    step = from_ckpt if from_ckpt is not None else from_name
    if step is None:
        raise SystemExit(f"STOP: no step for epoch {epoch} from the bundle or a "
                         f"head_ep{epoch}_step*.pt; refusing to write a guessed step column.")
    return step


def shard_is_complete(d):
    """A shard counts as done only if it FINISHED, not if it left a file behind.

    A child that crashed after its first trajectory flush leaves a trajectory.csv that looks like
    a result. Treating that as complete means the one shard that actually needs recovery is the
    one the tool skips, and the population check then fails for a reason that points nowhere.
    So completeness is read from the shard's own coverage line and the row count it implies.
    """
    lg, tj = os.path.join(d, "shard.log"), os.path.join(d, "trajectory.csv")
    if not (os.path.exists(lg) and os.path.exists(tj)):
        return False, "no log or no trajectory"
    cov = None
    try:
        with open(lg, errors="replace") as fh:
            for ln in fh:
                if ln.startswith("SHARD COVERAGE "):
                    cov = dict(kv.split("=", 1) for kv in ln.split()[2:] if "=" in kv)
    except OSError as e:
        return False, f"unreadable log ({e})"
    if not cov:
        return False, "no SHARD COVERAGE line (the child did not get that far)"
    try:
        want = int(cov["units_mine"]) * int(cov["nsample"])
    except (KeyError, ValueError):
        return False, "malformed coverage line"
    try:
        with open(tj) as fh:
            got = sum(1 for r in csv.DictReader(fh) if r.get("pae_loss"))
    except OSError as e:
        return False, f"unreadable trajectory ({e})"
    if got != want:
        return False, f"{got} pose row(s) against {want} the coverage line claims"
    return True, f"{got} pose(s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rid")
    ap.add_argument("epoch", type=int)
    ap.add_argument("--cards", default="", help="comma list; default = the run's --eval-cards")
    ap.add_argument("--verify-against", type=int, default=None,
                    help="shard index the PARENT launched successfully, to compare args against; "
                         "default = the lowest-numbered shard that already has an args.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="publish even when no parent-launched shard exists to compare the "
                         "rebuilt command against. Off by default: the comparison is the only "
                         "check that the hand-rebuilt command measures the same model.")
    a = ap.parse_args()

    run = os.path.join(ROOT, "training", "runs", a.rid)
    args = json.load(open(os.path.join(run, "args.json")))
    bundle = os.path.join(run, f"resume_ep{a.epoch}.pt")
    if not os.path.exists(bundle):
        raise SystemExit(f"STOP: no {bundle}; nothing to measure.")
    cards = [c.strip() for c in (a.cards or args.get("eval_cards") or "").split(",") if c.strip()]
    if len(cards) < 2:
        raise SystemExit(f"STOP: need the run's multi-card shard layout, got {cards}.")
    nsh = len(cards)
    base = os.path.join(run, f"val_loss_ep{a.epoch}")

    dirs = {i: os.path.join(base, f"shard{i}") for i in range(nsh)}
    done, why = {}, {}
    for i, d in dirs.items():
        ok, msg = shard_is_complete(d)
        why[i] = msg
        if ok:
            done[i] = d
    todo = [i for i in range(nsh) if i not in done]
    for i in range(nsh):
        print(f"  shard {i}: {'COMPLETE' if i in done else 'needs re-run'} -- {why[i]}")
    print(f"{a.rid} epoch {a.epoch}: {len(done)} shard(s) already complete {sorted(done)}, "
          f"re-running {todo} on card(s) {[cards[i] for i in todo]}")
    if not todo:
        print("nothing to re-run")
    ref_i = a.verify_against
    if ref_i is None:
        ref_i = next((i for i in sorted(done) if os.path.exists(
            os.path.join(dirs[i], "args.json"))), None)

    # A DRY RUN MUST NOT TOUCH THE FILESYSTEM. This used to rotate the failed shard.log before
    # testing the flag, so asking what the tool WOULD do destroyed the path the failure evidence
    # lived at -- and a later real run then found no shard.log to rotate and produced no second
    # copy. Printing comes first and returns; nothing below it runs. Caught in review.
    if a.dry_run:
        for i in todo:
            cmd = build_cmd(args, bundle, dirs[i], a.epoch) + \
                ["--manifest-shard", str(i), "--manifest-nshards", str(nsh)]
            print(f"\nshard {i} on card {cards[i]}:\n  " + " ".join(cmd))
        print("\n(dry run: nothing was launched and no file was moved)")
        return 0

    procs = []
    for i in todo:
        out = dirs[i]
        os.makedirs(out, exist_ok=True)
        # ROTATE EVERY OUTPUT OF THE FAILED ATTEMPT, not just its log. trajectory.csv is opened
        # with "a" by the child, so a shard that died after emitting rows would have this re-run
        # append a SECOND copy of them and the pooling would refuse the epoch for a duplicate
        # pose -- the same defect the trainer's retry had. They are rotated rather than deleted:
        # a partial trajectory is the evidence of where the attempt died.
        n = 0
        while os.path.exists(os.path.join(out, f"shard.log.failed{n or ''}")):
            n += 1
        sfx = f".failed{n or ''}"
        for old_p in ([os.path.join(out, f) for f in
                       ("shard.log", "trajectory.csv", "iptm_rows.csv")]
                      + sorted(glob.glob(os.path.join(out, "crop_sizes_rank*.csv")))):
            if os.path.exists(old_p):
                os.rename(old_p, old_p + sfx)
        lg = os.path.join(out, "shard.log")
        cmd = build_cmd(args, bundle, out, a.epoch) + \
            ["--manifest-shard", str(i), "--manifest-nshards", str(nsh)]
        fh = open(lg, "w")
        try:
            pr = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True,
                                  env=child_env(cards[i]))
        except Exception:
            fh.close()
            raise
        procs.append((i, out, lg, fh, pr))
        print(f"  shard {i} -> card {cards[i]}, pid {pr.pid}, log {lg}")

    bad = []
    for i, out, lg, fh, p in procs:
        p.wait()
        fh.close()
        if p.returncode != 0:
            tail = open(lg).read().strip().splitlines()[-6:]
            bad.append((i, p.returncode, tail))
    for i, rc, tail in bad:
        print(f"  shard {i} FAILED again (rc={rc}):\n    " + "\n    ".join(tail))
    if bad:
        raise SystemExit(f"STOP: {len(bad)} shard(s) failed again; no held-out point.")

    # SAME MODEL, SAME POPULATION -- asserted, not assumed.
    if ref_i is not None:
        ref = json.load(open(os.path.join(dirs[ref_i], "args.json")))
        for i, out, _lg, _fh, _p in procs:
            got = json.load(open(os.path.join(out, "args.json")))
            diff = {k for k in set(ref) | set(got)
                    if ref.get(k) != got.get(k)} - SHARD_FIELDS
            if diff:
                raise SystemExit(
                    f"STOP: re-run shard {i} was configured differently from parent-launched "
                    f"shard {ref_i} in {sorted(diff)}:\n  " + "\n  ".join(
                        f"{k}: parent {ref.get(k)!r} vs re-run {got.get(k)!r}"
                        for k in sorted(diff)))
        print(f"  args verified identical to parent-launched shard {ref_i} "
              f"(ignoring {sorted(SHARD_FIELDS)})")
    elif not a.allow_unverified:
        raise SystemExit(
            "STOP: no parent-launched shard has an args.json to verify the rebuilt command "
            "against, so there is nothing to prove this measured the same model and population "
            "as the epoch's surviving shards.\n"
            "  build_cmd() duplicates heldout_pae()'s construction by hand and that comparison is "
            "the ONLY thing that catches a divergence.\n"
            "  Pass --allow-unverified to publish anyway, and say so wherever the number is used.")
    else:
        print("  WARNING: --allow-unverified: the rebuilt command was NOT checked against a "
              "parent-launched shard. This number rests on build_cmd() being correct.")

    mean, npose = pool(dirs, nsh)
    print(f"\nepoch {a.epoch} VALLOSS: pae {mean:.4f}  poses {npose}   [RECOVERED]")

    # WRITE IT WHERE EVERY READER LOOKS. A recovered number that lives only in this script's
    # stdout is still a missing PAE value as far as the plots, EXPERIMENTS.md and every later
    # comparison are concerned. The trainer's own writer is append-only (it is shared by the
    # sharded and unsharded paths and documented as "Append one held-out point"), so appending
    # here cannot race a rewrite -- but it CAN duplicate, so the epoch is checked first.
    vp = os.path.join(run, "val_loss.csv")
    have = set()
    if os.path.exists(vp):
        with open(vp) as fh:
            for row in csv.DictReader(fh):
                try:
                    have.add(int(row["epoch"]))
                except (KeyError, TypeError, ValueError):
                    continue
    if a.epoch in have:
        print(f"  val_loss.csv already has epoch {a.epoch}; NOT appending a second row")
        return 0
    # THE STEP COLUMN IS DERIVED, NOT GUESSED. The bundle's own step is the authority; the
    # per-epoch cadence is only a cross-check, and a disagreement means the assumption is wrong.
    step = bundle_step(bundle, run, a.epoch)
    with open(vp, "a", newline="") as fh:
        csv.writer(fh).writerow([a.epoch, step, npose, f"{mean:.6f}",
                                 os.path.basename(bundle)])
    print(f"  appended to {vp}: epoch {a.epoch}, step {step}, {npose} poses, {mean:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
