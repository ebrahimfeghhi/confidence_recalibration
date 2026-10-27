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
import json
import os
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
        cmd += ["--emit-iptm-rows", os.path.join(os.path.dirname(out), "iptm_rows.csv")]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rid")
    ap.add_argument("epoch", type=int)
    ap.add_argument("--cards", default="", help="comma list; default = the run's --eval-cards")
    ap.add_argument("--verify-against", type=int, default=None,
                    help="shard index the PARENT launched successfully, to compare args against; "
                         "default = the lowest-numbered shard that already has an args.json")
    ap.add_argument("--dry-run", action="store_true")
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
    done = {i: d for i, d in dirs.items() if os.path.exists(os.path.join(d, "trajectory.csv"))}
    todo = [i for i in range(nsh) if i not in done]
    print(f"{a.rid} epoch {a.epoch}: {len(done)} shard(s) already complete {sorted(done)}, "
          f"re-running {todo} on card(s) {[cards[i] for i in todo]}")
    if not todo:
        print("nothing to re-run")
    ref_i = a.verify_against
    if ref_i is None:
        ref_i = next((i for i in sorted(done) if os.path.exists(
            os.path.join(dirs[i], "args.json"))), None)

    procs = []
    for i in todo:
        out = dirs[i]
        os.makedirs(out, exist_ok=True)
        # The failed attempt left a shard.log. Keep it -- it is the evidence of the failure --
        # under a .failed name, so the coverage scan reads only the new one.
        lg = os.path.join(out, "shard.log")
        if os.path.exists(lg):
            n = 0
            while os.path.exists(f"{lg}.failed{n or ''}"):
                n += 1
            os.rename(lg, f"{lg}.failed{n or ''}")
        cmd = build_cmd(args, bundle, out, a.epoch) + \
            ["--manifest-shard", str(i), "--manifest-nshards", str(nsh)]
        if a.dry_run:
            print(f"\nshard {i} on card {cards[i]}:\n  " + " ".join(cmd))
            continue
        fh = open(lg, "w")
        procs.append((i, out, lg, fh,
                      subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True,
                                       env=child_env(cards[i]))))
        print(f"  shard {i} -> card {cards[i]}, pid {procs[-1][4].pid}, log {lg}")
    if a.dry_run:
        return 0

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
    else:
        print("  WARNING: no parent-launched shard with an args.json to verify against")

    mean, npose = pool(dirs, nsh)
    print(f"\nepoch {a.epoch} VALLOSS: pae {mean:.4f}  poses {npose}   [RECOVERED]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
