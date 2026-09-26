"""A dead worker's claim is released when ITS fold is unscored, even if other folds of the complex share
its seed. CPU only.

    $PROTENIX_PYTHON diffusion_commit_clean_v2/tests/test_reconcile_claims.py

Every templated fold uses one seed, so a check keyed on (complex, seed) saw any scored fold of the
complex and kept every dead claim -- a fold killed mid-run was then skipped forever.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import folder     # noqa: E402

DEAD_PID = 2 ** 22 + 12345          # above the default pid_max, so never a live process


def claim(d, name):
    os.makedirs(f"{d}/{name}")
    with open(f"{d}/{name}/owner", "w") as fh:
        fh.write(f"{folder.socket.gethostname()} {DEAD_PID}")


def test_dead_unscored_claim_released_despite_shared_seed():
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
        print("ok  dead claim of the unscored fold released; the scored fold's claim kept")


if __name__ == "__main__":
    test_dead_unscored_claim_released_despite_shared_seed()
    print("ALL PASSED")
