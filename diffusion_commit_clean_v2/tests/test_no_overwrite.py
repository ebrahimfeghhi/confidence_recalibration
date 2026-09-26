"""No fold may overwrite another fold's poses. CPU only: the fold and the scorer are stubbed.

    $PROTENIX_PYTHON diffusion_commit_clean_v2/tests/test_no_overwrite.py

1. run_id() stays unique when the global `random` is re-seeded before every call, as Protenix does
   before every fold -- the bug that put 173 fixed-arm folds in 27 directories.
2. Two folds forced onto the SAME run directory: the second fails before folding and the first
   fold's poses are untouched.
3. Moving a pose onto an existing file raises instead of overwriting.
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


def test_run_id_survives_reseeding():
    ids = set()
    for _ in range(500):
        random.seed(430000)            # what seed_everything does before every fold
        ids.add(runmeta.run_id())
    assert len(ids) == 500, f"only {len(ids)} distinct ids from 500 re-seeded calls"
    print("ok  run_id: 500 distinct ids under random.seed(430000) before each call")


def make_ctx(tmp):
    jobs = f"{tmp}/jobs"
    os.makedirs(jobs)
    open(f"{jobs}/1abc-final-updated.json", "w").write("[{}]")
    A = types.SimpleNamespace(templates=None, claim_dir=f"{tmp}/claims", skip_by_template=False,
                              force=False, out=f"{tmp}/poses", csv=f"{tmp}/results.csv",
                              ncycle=10, nsample=5, dtype="bf16", keep_cifs=True)
    os.makedirs(A.claim_dir)
    ctx = types.SimpleNamespace(A=A, runner=None, configs=None, jobs=jobs,
                                ifaces={"1abc": {"1abc_AB"}}, rows_done=set(), units_done={})
    ctx.point_templates_at = lambda use_ns: None
    return ctx


def fake_fold(runner, configs, job, seed, name, n_cycle, n_sample, expect, dump_base, env):
    """Writes n_sample poses whose content names the fold, where Protenix would put them."""
    d = f"{dump_base}/{name}/seed_{seed}/predictions"
    os.makedirs(d, exist_ok=True)
    for i in range(n_sample):
        open(f"{d}/{name}_sample_{i}.cif", "w").write(f"pose of {name}\n")
        open(f"{d}/{name}_summary_confidence_sample_{i}.json", "w").write("{}")
    return 1.0, [4]


def test_shared_run_dir_fails_and_keeps_first():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_ctx(tmp)
        folder.fold = fake_fold
        folder.score_fold = lambda *a, **k: (1, "")
        folder.runmeta.run_id = lambda *a, **k: "collide000"      # force the collision
        folder.runmeta.write_run_json = lambda *a, **k: None
        folder.append_index = lambda *a, **k: None

        r1 = folder.run_unit(ctx, ("free", "1abc", "*", "F1", "430000", []))
        r2 = folder.run_unit(ctx, ("free", "1abc", "*", "F2", "430000", []))
        pred = f"{tmp}/poses/1abc/430000-collide000/predictions"
        content = open(f"{pred}/1abc_sample_0.cif").read().strip()
        assert r1 == "ran", r1
        assert r2 == "failed", f"second fold into an existing directory returned {r2!r}"
        assert content == "pose of 1abc_F1", f"first fold's pose was overwritten: {content!r}"
        assert not os.path.exists(f"{tmp}/claims/1abc_*_F2_430000".replace("*", "all")), \
            "failed unit kept its claim and would never be retried"
        print("ok  shared run dir: second fold refused, first fold's poses intact, claim released")


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


if __name__ == "__main__":
    real_run_id = runmeta.run_id
    test_run_id_survives_reseeding()
    test_move_refuses_overwrite()
    test_shared_run_dir_fails_and_keeps_first()
    runmeta.run_id = real_run_id
    print("ALL PASSED")
