"""folder.py regressions. CPU only: the fold and the scorer are stubbed.

    $PROTENIX_PYTHON diffusion_commit_clean_v2/tests/test_folder.py

Each test fails on the code before its fix.

1. run_id() stays unique when the global `random` is re-seeded before every call, as Protenix does
   before every fold -- the bug that put 173 fixed-arm folds in 27 directories.
2. Moving a pose onto an existing file raises instead of overwriting.
3. Two folds forced onto the SAME run directory: the second fails before folding and the first
   fold's poses are untouched.
4. A dead worker's claim is released when ITS fold is unscored, even though every templated fold of
   the complex shares its seed -- keyed on (complex, seed) it was kept, and the fold skipped forever.
"""
import os
import random
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import folder     # noqa: E402
import runmeta    # noqa: E402

DEAD_PID = 2 ** 22 + 12345          # above the default pid_max, so never a live process


def test_run_id_survives_reseeding():
    ids = set()
    for _ in range(500):
        random.seed(430000)            # what seed_everything does before every fold
        ids.add(runmeta.run_id())
    assert len(ids) == 500, f"only {len(ids)} distinct ids from 500 re-seeded calls"
    print("ok  run_id: 500 distinct ids under random.seed(430000) before each call")


def test_move_refuses_overwrite():
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
    """Writes n_sample poses whose content names the fold, where Protenix would put them."""
    d = f"{dump_base}/{name}/seed_{seed}/predictions"
    os.makedirs(d, exist_ok=True)
    for i in range(n_sample):
        open(f"{d}/{name}_sample_{i}.cif", "w").write(f"pose of {name}\n")
        open(f"{d}/{name}_summary_confidence_sample_{i}.json", "w").write("{}")
    return 1.0, [4]


def test_shared_run_dir_fails_and_keeps_first():
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
                                      dtype="bf16", keep_cifs=True)
            os.makedirs(A.claim_dir)
            ctx = types.SimpleNamespace(A=A, runner=None, configs=None, jobs=jobs,
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
    def claim(d, name):
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


if __name__ == "__main__":
    test_run_id_survives_reseeding()
    test_move_refuses_overwrite()
    test_shared_run_dir_fails_and_keeps_first()
    test_dead_claim_released_despite_shared_seed()
    print("ALL PASSED")
