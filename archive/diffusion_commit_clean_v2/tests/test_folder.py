#!/usr/bin/env python
"""Regression tests for folder.py -- each fails on the code before its fix.

SUMMARY
    CPU only: the fold and the scorer are stubbed, so these run in seconds and never touch a GPU.

      1. run_id() stays unique when the global `random` is re-seeded before every call, as Protenix
         does before every fold -- the bug that put 173 fixed-arm folds in 27 directories.
      2. Moving a pose onto an existing file raises instead of overwriting.
      3. Two folds forced onto the SAME run directory: the second fails before folding, and the first
         fold's poses are untouched.
      4. A dead worker's claim is released when ITS fold is unscored, even though every templated fold
         of the complex shares its seed -- keyed on (complex, seed) it was kept and the fold skipped
         forever.
      5. --overlap-scoring: a unit is scored while the next one folds, no two units are ever scored at
         once, and every unit still ends "ran" with its poses in place.
      6. --overlap-scoring: a scoring exception fails the unit and releases its claim.

INPUTS
    none

OUTPUTS
    stdout: one "ok" line per test, then ALL PASSED; an AssertionError names the first failure

EXAMPLE
    $PROTENIX_PYTHON diffusion_commit_clean_v2/tests/test_folder.py

FUNCTIONS
    test_run_id_survives_reseeding():                1
    test_move_refuses_overwrite():                   2
    test_shared_run_dir_fails_and_keeps_first():     3, with fake_fold() standing in for Protenix
    test_dead_claim_released_despite_shared_seed():  4
    test_overlap_scoring():                          5
    test_overlap_scoring_failure_releases_claim():   6
    stubbed():                                       folder with fold, scorer and run.json stubbed
"""
import glob
import os
import random
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import folder     # noqa: E402
import runmeta    # noqa: E402

DEAD_PID = 2 ** 22 + 12345          # above the default pid_max, so never a live process


def test_run_id_survives_reseeding():
    """in: nothing.  out: passes, or AssertionError if re-seeding makes run ids repeat"""
    ids = set()
    for _ in range(500):
        random.seed(430000)            # what seed_everything does before every fold
        ids.add(runmeta.run_id())
    assert len(ids) == 500, f"only {len(ids)} distinct ids from 500 re-seeded calls"
    print("ok  run_id: 500 distinct ids under random.seed(430000) before each call")


def test_move_refuses_overwrite():
    """in: nothing.  out: passes, or AssertionError if an existing pose file is overwritten"""
    with tempfile.TemporaryDirectory() as tmp:
        a, b = f"{tmp}/a", f"{tmp}/b"
        open(a, "w").write("new")
        open(b, "w").write("old")
        try:
            folder._move_new(a, b)
        except FileExistsError:
            assert open(b).read() == "old"
            print("ok  _move_new: refuses to overwrite, existing file untouched")
            return
        raise AssertionError("_move_new overwrote an existing file")


def fake_fold(runner, configs, job, seed, name, n_cycle, n_sample, expect, dump_base, env):
    """in: fold() arguments.  out: (seconds, template counts); writes n_sample poses naming the fold"""
    d = f"{dump_base}/{name}/seed_{seed}/predictions"
    os.makedirs(d, exist_ok=True)
    for i in range(n_sample):
        open(f"{d}/{name}_sample_{i}.cif", "w").write(f"pose of {name}\n")
        open(f"{d}/{name}_summary_confidence_sample_{i}.json", "w").write("{}")
    return 1.0, [4]


def test_shared_run_dir_fails_and_keeps_first():
    """in: nothing.  out: passes, or AssertionError if a second fold into a used directory runs"""
    stubs = {"fold": fake_fold, "score_fold": lambda *a, **k: (1, ""),
             "append_index": lambda *a, **k: None}
    saved = {k: getattr(folder, k) for k in stubs}
    saved_rm = (runmeta.run_id, runmeta.write_run_json)
    try:
        for k, v in stubs.items():
            setattr(folder, k, v)
        runmeta.run_id = lambda *a, **k: "collide000"      # force the collision
        runmeta.write_run_json = lambda *a, **k: None
        with tempfile.TemporaryDirectory() as tmp:
            jobs = f"{tmp}/jobs"
            os.makedirs(jobs)
            open(f"{jobs}/1abc-final-updated.json", "w").write("[{}]")
            A = types.SimpleNamespace(templates=None, claim_dir=f"{tmp}/claims",
                                      skip_by_template=False, force=False, out=f"{tmp}/poses",
                                      csv=f"{tmp}/results.csv", ncycle=10, nsample=5,
                                      dtype="bf16", keep_cifs=True, overlap_scoring=False)
            os.makedirs(A.claim_dir)
            ctx = types.SimpleNamespace(A=A, runner=None, configs=None, jobs=jobs, free_templates=True,
                                        ifaces={"1abc": {"1abc_AB"}}, rows_done=set(),
                                        units_done={}, point_templates_at=lambda use_ns: None)
            r1 = folder.run_unit(ctx, ("free", "1abc", "*", "F1", "430000", []))
            r2 = folder.run_unit(ctx, ("free", "1abc", "*", "F2", "430000", []))
            pose = f"{tmp}/poses/1abc/430000-collide000/predictions/1abc_sample_0.cif"
            content = open(pose).read().strip()
            assert r1 == "ran", r1
            assert r2 == "failed", f"second fold into an existing directory returned {r2!r}"
            assert content == "pose of 1abc_F1", f"first fold's pose was overwritten: {content!r}"
            assert not os.path.exists(f"{tmp}/claims/1abc_all_F2_430000"), \
                "failed unit kept its claim and would never be retried"
    finally:
        for k, v in saved.items():
            setattr(folder, k, v)
        runmeta.run_id, runmeta.write_run_json = saved_rm
    print("ok  shared run dir: second fold refused, first fold's poses intact, claim released")


def test_dead_claim_released_despite_shared_seed():
    """in: nothing.  out: passes, or AssertionError if the dead claim of the unscored fold is kept"""
    def claim(d, name):
        """in: a claim dir and a claim name.  out: the claim, owned by a pid that cannot be alive"""
        os.makedirs(f"{d}/{name}")
        with open(f"{d}/{name}/owner", "w") as fh:
            fh.write(f"{folder.socket.gethostname()} {DEAD_PID}")

    with tempfile.TemporaryDirectory() as tmp:
        cd, csv_path = f"{tmp}/claims", f"{tmp}/results.csv"
        claim(cd, "1abc_1abc_AB_A00_430000")      # scored
        claim(cd, "1abc_1abc_AB_A01_430000")      # killed mid-fold: no rows
        with open(csv_path, "w") as fh:
            fh.write("iface,seed,arm,donor_pose,tmpl_chains,sample,dockq,iptm\n")
            for s in range(5):
                fh.write(f"1abc_AB,430000,A00,/x.cif,A+B,{s},0.5,0.8\n")
        folder.reconcile_claims(cd, csv_path, {"1abc": {"1abc_AB"}})
        left = sorted(os.listdir(cd))
        assert left == ["1abc_1abc_AB_A00_430000"], f"claims left: {left}"
    print("ok  reconcile_claims: the unscored fold's dead claim released, the scored one kept")


class stubbed:
    """in: a scorer stand-in.  out: a context manager with folder's fold, scoring and bookkeeping stubbed

    The fold sleeps and writes poses, so timings can be compared; everything is restored on exit.
    """

    def __init__(self, score, fold_s=0.3):
        self.score, self.fold_s, self.events = score, fold_s, []

    def __enter__(self):
        def fold(runner, configs, job, seed, name, n_cycle, n_sample, expect, dump_base, env):
            self.events.append(("fold", name, time.time()))
            time.sleep(self.fold_s)
            out = fake_fold(runner, configs, job, seed, name, n_cycle, n_sample, expect, dump_base, env)
            self.events.append(("fold_end", name, time.time()))
            return out
        self.saved = {k: getattr(folder, k) for k in ("fold", "score_fold", "append_index")}
        self.saved_rm = runmeta.write_run_json
        folder.fold, folder.score_fold = fold, self.score
        folder.append_index = lambda *a, **k: None
        runmeta.write_run_json = lambda *a, **k: None
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(folder, k, v)
        runmeta.write_run_json = self.saved_rm


def overlap_ctx(tmp):
    """in: a temp dir.  out: a folder context with --overlap-scoring on and a free-fold job json"""
    jobs = f"{tmp}/jobs"
    os.makedirs(jobs)
    open(f"{jobs}/1abc-final-updated.json", "w").write("[{}]")
    A = types.SimpleNamespace(templates=None, claim_dir=f"{tmp}/claims", skip_by_template=False,
                              out=f"{tmp}/poses", csv=f"{tmp}/results.csv", ncycle=10, nsample=5,
                              dtype="bf16", keep_cifs=True, overlap_scoring=True)
    os.makedirs(A.claim_dir)
    return types.SimpleNamespace(A=A, runner=None, configs=None, jobs=jobs, free_templates=True,
                                 ifaces={"1abc": {"1abc_AB"}}, rows_done=set(), units_done={},
                                 point_templates_at=lambda use_ns: None)


def test_overlap_scoring():
    """in: nothing.  out: passes, or AssertionError if scoring does not overlap folding, or overlaps
    itself, or a unit ends anything but "ran" with its poses in place"""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = overlap_ctx(tmp)

        def score(ctx_, pdb, target, tag, seed, rdir, name, want, poses):
            st.events.append(("score", name, time.time()))
            time.sleep(0.5)
            st.events.append(("score_end", name, time.time()))
            for i in want:
                ctx_.rows_done.add((i, str(seed), tag))
            return len(want), ""

        with stubbed(score) as st:
            units = [("free", "1abc", "*", f"F{k}", "430000", []) for k in range(3)]
            tally = folder.run_queue(ctx, units)
        assert tally == {"ran": 3}, tally
        t = {(e, n): ts for e, n, ts in st.events}
        # scoring of unit k overlaps the fold of unit k+1
        assert t[("fold", "1abc_F1")] < t[("score_end", "1abc_F0")], "no overlap"
        # never two units scored at once
        spans = sorted((t[("score", f"1abc_F{k}")], t[("score_end", f"1abc_F{k}")]) for k in range(3))
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "scorings overlapped"
        poses = glob.glob(f"{tmp}/poses/1abc/*/predictions/1abc_sample_0.cif")
        assert len(poses) == 3, poses
        # a scored unit keeps its claim -- the claim is what marks it done for other workers
        assert len(os.listdir(ctx.A.claim_dir)) == 3, os.listdir(ctx.A.claim_dir)
    print("ok  overlap scoring: scored during the next fold, one at a time, all 3 ran with poses kept")


def test_overlap_scoring_failure_releases_claim():
    """in: nothing.  out: passes, or AssertionError if a scoring exception is not a failed unit with
    its claim released"""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = overlap_ctx(tmp)

        def score(*a, **k):
            raise RuntimeError("scorer crashed")

        with stubbed(score):
            tally = folder.run_queue(ctx, [("free", "1abc", "*", "F0", "430000", [])])
        assert tally == {"failed": 1}, tally
        assert os.listdir(ctx.A.claim_dir) == [], os.listdir(ctx.A.claim_dir)
    print("ok  overlap scoring: a scoring exception fails the unit and releases its claim")


def test_scorer_launch_retried():
    """in: nothing.  out: passes, or AssertionError if a scorer that fails to START is not relaunched,
    or if a launch failing every time is not raised after SPAWN_TRIES"""
    real, sleep, calls = folder.subprocess.run, folder.time.sleep, []

    def flaky(cmd, **k):
        calls.append(cmd)
        if len(calls) == 1:
            raise OSError(14, "Bad address")
        return "ran"

    def broken(cmd, **k):
        calls.append(cmd)
        raise OSError(14, "Bad address")

    folder.time.sleep = lambda s: None
    try:
        folder.subprocess.run = flaky
        assert folder.spawn(["scorer"]) == "ran" and len(calls) == 2, calls
        calls.clear()
        folder.subprocess.run = broken
        try:
            folder.spawn(["scorer"])
            raise AssertionError("a launch failing every time was not raised")
        except OSError:
            pass
        assert len(calls) == folder.SPAWN_TRIES, calls
    finally:
        folder.subprocess.run, folder.time.sleep = real, sleep
    assert folder.subprocess._USE_VFORK is False
    print("ok  scorer launch: retried after OSError, raised after SPAWN_TRIES, vfork off")


if __name__ == "__main__":
    test_run_id_survives_reseeding()
    test_move_refuses_overwrite()
    test_shared_run_dir_fails_and_keeps_first()
    test_dead_claim_released_despite_shared_seed()
    test_overlap_scoring()
    test_overlap_scoring_failure_releases_claim()
    test_scorer_launch_retried()
    print("ALL PASSED")
