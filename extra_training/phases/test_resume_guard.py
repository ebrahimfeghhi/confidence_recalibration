"""The resume guard must refuse only what cannot be reconstructed, and record everything else.

Run: python -I phases/test_resume_guard.py

WHY THIS EXISTS. Resuming Arm B's epoch-10 bundle to add a cosine decay took FOUR failed launches
and found no real problem in any of them: a field recorded in neither the bundle nor its args.json
was a hard STOP, two flags a PAE-only arm never reads compared "unset" against "unset", and the
name that got past the first stop became a "stale override" error on the next bundle. The weights
and the AdamW moments were intact every time. So the guard now ALLOWS a changed lr, loss weight or
rank setting with a loud grouped warning and a recorded resume_drift entry, and hard-fails only
where the state cannot be restored: not a resume bundle, no moments, a missing trainable tensor, or
moments whose count or shape does not match this run's trainable set.

A test that asserted the warning text would pass while the resume died, so this one runs the REAL
region: it is extracted from phase_k_train.py by its first and last marker lines, dedented once and
exec'd against a two-parameter model and real AdamW state. The optimiser restore is checked by
reading exp_avg_sq back out, not by trusting a log line that once claimed "298 tensor(s) restored"
on a pass that restored nothing.

Nothing here touches training/runs: every bundle is written to a fresh temp directory.
"""
import copy
import datetime as _dt
import json
import math
import os
import sys
import tempfile
import textwrap
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "phase_k_train.py")
WORK = tempfile.mkdtemp(prefix="resume_guard_")

LINES = open(SRC).read().splitlines(keepends=True)
START = next(i for i, l in enumerate(LINES) if l.rstrip("\n") == "    if a.resume:")
END = next(i for i, l in enumerate(LINES)
           if i > START and "# THE L2-SP ANCHOR IS THE SHIPPED HEAD" in l)
REGION = textwrap.dedent("".join(LINES[START:END]))
print(f"extracted guard region: source lines {START + 1}..{END} ({END - START} lines)\n")


class Args:
    """argparse-like, with a type-neutral default for every flag the region reads."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, k):
        return 0


def make_model():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False), torch.nn.Linear(3, 2, bias=False))
    names = [n for n, _ in m.named_parameters()]
    opt = torch.optim.AdamW([p for _, p in m.named_parameters()], lr=1e-3)
    # ONE REAL STEP, so exp_avg / exp_avg_sq actually exist and have real shapes.
    m(torch.randn(2, 4)).sum().backward()
    opt.step()
    opt.zero_grad()
    return m, opt, sorted(names)


def base_args(**over):
    kw = dict(resume="", resume_override="", resume_allow_drift=False, resume_strict=False,
              resume_note="because the schedule changed", loss_only=False, epochs=20,
              out=os.path.join(WORK, "out"), train_only="", seed=1, accum=30, nsample=5,
              crop_size=3400, lr=1e-3, lr_final=0.0, warmup_steps=0, alpha_confidence=1.0,
              alpha_pae=1.0, drop_pde=False, clip_norm=1.0, rank_loss=False, alpha_rank=1.0,
              rank_pool="max", rel_labels="", rel_standardize=False, rel_sd_weight=False,
              rel_g_weight=0.0, pose_readout_prenorm=False, pose_readout_post_zero=False,
              rel_g_sigmoid=None, rel_std_floor=None, rank_form="rel", rank_accept=0.23,
              rank_cross_seed=False, rank_structs_per_step=2, lora_keep_trainable=False,
              l2_sp=0.0, l2_sp_exclude="", lr_decay_start_epoch=0.0, lr_decay_end_epoch=0.0,
              lr_decay_shape="cosine", lr_steps="", pair_dropout=-1.0, token_keep_union=-1.0,
              token_keep_other=-1.0, crop_contact=5.0, crop_all_poses=True,
              pae_scored_pairs="", pae_only=True, lean_single=False, pose_inblock=False,
              pose_inblock_pae=False, pose_inblock_block=-1, frame_direction=False,
              pose_perblock=True, pose_perblock_tied=False, pose_perblock_share=False,
              pose_perblock_copy=False, lora_rank=0, lora_alpha=0.0, lora_targets="",
              lora_dropout=0.0, epoch_cover_all=False, pose_batch=1, epoch_balance=False,
              pae_row_frac=0.0, pae_row_mass=0.8, pae_row_floor=10, manifest_skip="")
    kw.update(over)
    os.makedirs(kw["out"], exist_ok=True)
    return Args(**kw)


def write_bundle(name, a, model, opt, guard=None, drop_from_args=(), args_over=None,
                 opt_sd=None, head=None, raw=None):
    """A bundle plus the args.json beside it, as a real run directory looks."""
    d = os.path.join(WORK, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "resume_ep3.pt")
    if raw is not None:
        torch.save(raw, p)
    else:
        torch.save({"head": head if head is not None
                    else {n: q.detach().cpu() for n, q in model.named_parameters()},
                    "opt": opt_sd if opt_sd is not None else copy.deepcopy(opt.state_dict()),
                    "epoch": 3, "step": 300,
                    "guard": dict(guard or {})}, p)
    rec = {k: v for k, v in vars(a).items()}
    rec.update(args_over or {})
    for k in drop_from_args:
        rec.pop(k, None)
    json.dump(rec, open(os.path.join(d, "args.json"), "w"), indent=1, default=str)
    return p


def run(label, a, model, opt, trainable, world=3, expect="proceed"):
    ns = {"a": a, "model": model, "opt": opt, "trainable": list(trainable), "world": world,
          "is_main": True, "_manifest_digest": "deadbeef", "prefixes": ["head"],
          "start_epoch": 0, "start_step": 0, "_drift_history": [], "_resume_drift": None,
          "os": os, "json": json, "math": math, "time": time, "torch": torch, "_dt": _dt,
          "copy": copy, "sys": sys}
    print("=" * 78)
    print(f"CASE: {label}")
    print("-" * 78)
    try:
        exec(compile(REGION, "<guard-region>", "exec"), ns)
    except SystemExit as e:
        got = "hard-fail"
        print(f"SystemExit: {e}")
    else:
        got = "proceed"
        print(f"-> proceeded. start_epoch={ns['start_epoch']} start_step={ns['start_step']}")
        print(f"-> _resume_drift = {json.dumps(ns['_resume_drift'], indent=1, default=str)}")
    verdict = "OK" if got == expect else "UNEXPECTED"
    print(f"[{verdict}] expected {expect}, got {got}\n")
    return got == expect, ns


def main():
    results = []

    # 1  A SCHEDULE FORK PLUS AN UNRECORDED FIELD -- the Arm B case, which cost four launches.
    m, o, tr = make_model()
    a = base_args(lr_final=1e-5, lr_decay_start_epoch=10.0, lr_decay_end_epoch=20.0)
    a.resume = write_bundle("sched", a, m, o, guard={},
                            drop_from_args=("world",),
                            args_over={"lr_final": 0.0, "lr_decay_start_epoch": 0.0,
                                       "lr_decay_end_epoch": 0.0, "rel_g_sigmoid": None,
                                       "rel_std_floor": None})
    ok, ns = run("schedule drift (lr_final, decay window) + `world` recorded nowhere", a, m, o, tr)
    d = ns["_resume_drift"] or {}
    ok = ok and sorted(d.get("schedule_objective", [])) == ["lr_decay_end_epoch",
                                                            "lr_decay_start_epoch", "lr_final"]
    ok = ok and "world" in d.get("not_compared", []) and d.get("batch_composition") == []
    # the two unset-spelling fields must NOT appear anywhere in the drift
    ok = ok and not ({"rel_g_sigmoid", "rel_std_floor"} & set(d.get("fields", {})))
    print(f"  [{'OK' if ok else 'UNEXPECTED'}] grouped as schedule-only, world unverified, "
          f"rel_* not reported\n")
    results.append(("schedule fork proceeds, grouped, recorded", ok))
    # and it landed in resume_drift.json
    dp = os.path.join(a.out, "resume_drift.json")
    land = os.path.exists(dp) and json.load(open(dp))[0]["fields"].get("lr_final") is not None
    print(f"  [{'OK' if land else 'UNEXPECTED'}] resume_drift.json written at {dp}\n")
    results.append(("resume_drift.json written", land))

    # 2  A BATCH-COMPOSITION FORK -- allowed, but in the loudest group.
    m, o, tr = make_model()
    a = base_args()
    a.resume = write_bundle("batch", a, m, o, guard={"world": 3, "accum": 30, "seed": 1},
                            args_over={"rel_g_sigmoid": None, "rel_std_floor": None})
    ok, ns = run("world 3 -> 2 (batch composition)", a, m, o, tr, world=2)
    d = ns["_resume_drift"] or {}
    ok = ok and d.get("batch_composition") == ["world"] and d.get("schedule_objective") == []
    print(f"  [{'OK' if ok else 'UNEXPECTED'}] world reported in the loudest group\n")
    results.append(("batch-composition fork proceeds in the hot group", ok))

    # 3  --resume-strict RESTORES THE OLD REFUSAL.
    m, o, tr = make_model()
    a = base_args(lr_final=1e-5, resume_strict=True)
    a.resume = write_bundle("strict", a, m, o, guard={},
                            args_over={"lr_final": 0.0, "rel_g_sigmoid": None,
                                       "rel_std_floor": None})
    ok, _ = run("same schedule drift under --resume-strict", a, m, o, tr, expect="hard-fail")
    results.append(("--resume-strict still refuses unnamed drift", ok))

    # 4  AND --resume-override STILL PERMITS IT UNDER --resume-strict.
    m, o, tr = make_model()
    a = base_args(lr_final=1e-5, resume_strict=True, resume_override="lr_final")
    a.resume = write_bundle("strict2", a, m, o, guard={},
                            args_over={"lr_final": 0.0, "rel_g_sigmoid": None,
                                       "rel_std_floor": None})
    ok, _ = run("--resume-strict --resume-override lr_final", a, m, o, tr)
    results.append(("--resume-override permits it under --resume-strict", ok))

    # 5  A STALE OVERRIDE IS NO LONGER AN ERROR (Arm B's fourth failed launch).
    m, o, tr = make_model()
    a = base_args(resume_override="world,lr_final")
    a.resume = write_bundle("stale", a, m, o, guard={"world": 3},
                            args_over={"rel_g_sigmoid": None, "rel_std_floor": None})
    ok, _ = run("--resume-override names fields that do not differ (the 'stale override' case)",
                a, m, o, tr)
    results.append(("stale override no longer refuses", ok))

    # 6  A MISSING --resume-note NO LONGER REFUSES.
    m, o, tr = make_model()
    a = base_args(lr_final=1e-5, resume_note="")
    a.resume = write_bundle("nonote", a, m, o, guard={},
                            args_over={"lr_final": 0.0, "rel_g_sigmoid": None,
                                       "rel_std_floor": None})
    ok, ns = run("drift with an empty --resume-note", a, m, o, tr)
    results.append(("empty --resume-note no longer refuses", ok))

    # 7  WEIGHTS-ONLY CHECKPOINT -> a named refusal, not KeyError: 'epoch'.
    m, o, tr = make_model()
    a = base_args()
    a.resume = write_bundle("weightsonly", a, m, o,
                            raw={n: q.detach().cpu() for n, q in m.named_parameters()})
    ok, _ = run("head_ep10_step1450.pt (weights only)", a, m, o, tr, expect="hard-fail")
    results.append(("weights-only checkpoint named as not-a-bundle", ok))

    # 8  NO ADAMW MOMENTS -> hard fail.
    m, o, tr = make_model()
    a = base_args()
    a.resume = write_bundle("nomoments", a, m, o, opt_sd={"state": {},
                                                          "param_groups": [{"params": [0, 1]}]},
                            args_over={"rel_g_sigmoid": None, "rel_std_floor": None})
    ok, _ = run("bundle with an empty optimiser state", a, m, o, tr, expect="hard-fail")
    results.append(("missing AdamW moments hard-fails", ok))

    # 9  SHAPE-MISMATCHED MOMENTS -> hard fail (the thing torch does NOT check).
    m, o, tr = make_model()
    a = base_args()
    bad_sd = copy.deepcopy(o.state_dict())
    bad_sd["state"][0]["exp_avg"] = torch.zeros(7, 9)
    a.resume = write_bundle("shape", a, m, o, opt_sd=bad_sd,
                            args_over={"rel_g_sigmoid": None, "rel_std_floor": None})
    ok, _ = run("AdamW exp_avg saved at (7,9) for a (3,4) parameter", a, m, o, tr,
                expect="hard-fail")
    results.append(("shape-mismatched moments hard-fail", ok))

    # 10 COUNT-MISMATCHED MOMENTS -> hard fail.
    m, o, tr = make_model()
    a = base_args()
    few = copy.deepcopy(o.state_dict())
    few["param_groups"][0]["params"] = [0]
    few["state"].pop(1, None)
    a.resume = write_bundle("count", a, m, o, opt_sd=few,
                            args_over={"rel_g_sigmoid": None, "rel_std_floor": None})
    ok, _ = run("AdamW state for 1 parameter, optimiser holds 2", a, m, o, tr, expect="hard-fail")
    results.append(("count-mismatched moments hard-fail", ok))

    # 11 A MISSING TRAINABLE TENSOR -> hard fail (unchanged).
    m, o, tr = make_model()
    a = base_args()
    a.resume = write_bundle("missing", a, m, o,
                            head={"0.weight": torch.zeros(3, 4)},
                            args_over={"rel_g_sigmoid": None, "rel_std_floor": None})
    ok, _ = run("bundle missing one trainable tensor", a, m, o, tr, expect="hard-fail")
    results.append(("missing trainable tensor hard-fails", ok))

    # 12 A CLEAN CONTINUATION STAYS SILENT AND STAMPS NOTHING.
    m, o, tr = make_model()
    a = base_args()
    a.resume = write_bundle("clean", a, m, o, guard={"world": 3},
                            args_over={"rel_g_sigmoid": None, "rel_std_floor": None})
    ok, ns = run("every setting matches (clean continuation)", a, m, o, tr)
    ok = ok and ns["_resume_drift"] is None and ns["start_epoch"] == 3
    print(f"  [{'OK' if ok else 'UNEXPECTED'}] no drift recorded, start_epoch 3\n")
    results.append(("clean continuation records nothing", ok))

    # 13 THE OPTIMISER REALLY IS RESTORED on the allowed fork (not just not-refused).
    m, o, tr = make_model()
    a = base_args(lr_final=1e-5)
    saved = copy.deepcopy(o.state_dict())
    a.resume = write_bundle("restore", a, m, o, opt_sd=saved,
                            args_over={"lr_final": 0.0, "rel_g_sigmoid": None,
                                       "rel_std_floor": None})
    m2, o2, _ = make_model()
    for st in o2.state.values():          # wipe the moments so a restore is visible
        st["exp_avg"].zero_()
        st["exp_avg_sq"].zero_()
    ok, ns = run("the fork's AdamW moments are actually loaded", a, m2, o2, tr)
    live = list(o2.state.values())[0]
    nz = float(live["exp_avg_sq"].abs().sum()) > 0
    print(f"  [{'OK' if nz else 'UNEXPECTED'}] exp_avg_sq restored non-zero "
          f"({float(live['exp_avg_sq'].abs().sum()):.3e})\n")
    results.append(("moments restored on an allowed fork", ok and nz))

    # 14 THE --loss-only CHILD STILL RUNS, which is the one path that can break a LIVE arm. The
    # trainer spawns a held-out measurement once per epoch and that child reads this file FRESH
    # FROM DISK, so anything added here reaches a running trainer the moment the file is saved --
    # the comment inside the region says a mandatory flag once did exactly that. A measurement
    # legitimately differs from its bundle in the fold, the selection, the seeds and the lr, takes
    # no step and writes no checkpoint, so it must proceed, say one line rather than the banner,
    # and stamp nothing.
    m, o, tr = make_model()
    a = base_args(loss_only=True, resume_note="", lr=5e-4, lr_final=1e-5, epochs=1)
    a.resume = write_bundle("lossonly", a, m, o, guard={"lr": 1e-3, "world": 3},
                            args_over={"lr": 1e-3, "lr_final": 0.0, "rel_g_sigmoid": None,
                                       "rel_std_floor": None})
    ok, ns = run("--loss-only child with drift, no note, bundle epoch 3 > --epochs 1", a, m, o, tr)
    # epoch reset to 0 so the single pass runs, nothing stamped, no banner
    ok = ok and ns["start_epoch"] == 0 and ns["_resume_drift"] is None
    print(f"  [{'OK' if ok else 'UNEXPECTED'}] measurement proceeds, epoch reset to 0, "
          f"nothing stamped\n")
    results.append(("--loss-only measurement unaffected", ok))

    # 15 AND THE DRIFT REACHES WHERE RESULTS ARE READ. The record is built inside the region, so
    # the region test above cannot see the two places it is CONSUMED -- and a printed warning in a
    # 25k-line run.log is the failure mode this whole patch exists to fix. These two lines are the
    # stamp; if either is deleted the drift silently stops travelling with the numbers.
    src = open(SRC).read()
    for what, needle in (("args.json carries resume_drift",
                          '_args_rec["resume_drift"] = _resume_drift'),
                         ("every metric_*.json carries resume_drift",
                          'm["resume_drift"] = _resume_drift')):
        hit = needle in src
        print(f"  [{'OK' if hit else 'UNEXPECTED'}] {what}")
        results.append((what, hit))
    print()

    print("=" * 78)
    bad = [n for n, k in results if not k]
    for n, k in results:
        print(f"  {'PASS' if k else 'FAIL'}  {n}")
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
