"""Regression tests for the record reader, one per bug a review found in it.

RUN: python -I run_logging/test_records.py    (no GPU, no run directory, ~1 second)

Every case here is a bug that shipped and was caught by someone reading the code rather than by
anything in the repo. The reader is the only thing standing between the trajectory and a chart
someone makes decisions from, and it has now been wrong three times: it plotted one pose of 255 as
the batch, it spliced an epoch's trailing poses into the next epoch, and it ignored batch_loss so
the total line disagreed with its own components. These lock each of those down.
"""
import csv, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import records as R

COLS = (["wall_s", "step", "unit_i", "pdb", "unit", "pose", "lr"]
        + ["plddt_loss", "pde_loss", "pae_loss", "resolved_loss", "loss"]
        + ["batch_" + k for k in ("plddt_loss", "pde_loss", "pae_loss", "resolved_loss", "loss")]
        + ["batch_n", "partial_step", "step_ok", "grad_norm", "outside_head", "epoch",
           "step_taken", "nonfinite_grad"])


def row(**kw):
    r = {c: "" for c in COLS}
    r.update({"wall_s": "0", "step": "0", "unit_i": "0", "pdb": "x", "unit": "u", "pose": "0",
              "lr": "1e-3", "grad_norm": "1.0", "outside_head": "-1", "epoch": "1",
              "step_taken": "0", "nonfinite_grad": "0", "partial_step": "0", "step_ok": "1"})
    r.update({k: str(v) for k, v in kw.items()})
    return r


def read(rows):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "trajectory.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    return R.read_run(d)


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got}\n        want {want}")
    return ok


def main():
    ok = []
    print("THE TOTAL IS batch_loss, NOT THE STEPPING POSE'S OWN LOSS")
    # logged batch total 3.0, stepping pose's own loss 9.0: the chart must read 3.0
    rows = [row(loss=9.0, plddt_loss=9.0, batch_loss=3.0, batch_plddt_loss=1.0,
                batch_pae_loss=1.0, batch_pde_loss=0.5, batch_resolved_loss=0.5,
                batch_n=255, step_taken=1)]
    s = read(rows)["steps"]
    ok.append(check("total is the batch value", s[0]["loss"], 3.0))
    ok.append(check("components sum to the total", round(s[0]["raw"], 6), 3.0))

    print("A REFUSED STEP IS NOT A STEP")
    rows = [row(batch_loss=5.0, batch_n=255, step_taken=1, step_ok=0, nonfinite_grad=1),
            row(batch_loss=4.0, batch_n=255, step_taken=1)]
    s = read(rows)["steps"]
    ok.append(check("refused step excluded", [p["loss"] for p in s], [4.0]))

    print("A PARTIAL STEP STILL HAS A TOTAL")
    # the partial row carries only batch_* -- its per-pose loss field is empty by construction
    rows = [row(batch_loss=7.0, batch_n=170, step_taken=1, partial_step=1, loss="")]
    s = read(rows)["steps"]
    ok.append(check("partial step kept, with its batch total", [p["loss"] for p in s], [7.0]))
    ok.append(check("partial step flagged", s[0]["partial"], True))
    ok.append(check("its pose count is the reduced one", s[0]["n_poses"], 170))

    print("WITHOUT BATCH COLUMNS, FALL BACK TO RANK-0 WINDOW MEANS, NOT THE LAST POSE")
    rows = ([row(loss=10.0, plddt_loss=10.0) for _ in range(2)]
            + [row(loss=1.0, plddt_loss=1.0, step_taken=1)])
    s = read(rows)["steps"]
    ok.append(check("mean of the window, not its last pose", round(s[0]["loss"], 4), 7.0))
    ok.append(check("marked reconstructed, not exact", s[0].get("batch_n"), None))

    print("AN EPOCH BOUNDARY CLOSES A WINDOW (the trainer steps there and logs no row)")
    rows = ([row(loss=10.0, epoch=1) for _ in range(2)] + [row(loss=10.0, epoch=1, step_taken=1)]
            + [row(loss=1.0, epoch=1)]                      # epoch 1's trailing poses
            + [row(loss=20.0, epoch=2) for _ in range(2)] + [row(loss=20.0, epoch=2, step_taken=1)])
    s = read(rows)["steps"]
    ok.append(check("three points, not two", len(s), 3))
    ok.append(check("epoch 2 is not contaminated by epoch 1's tail",
                    [round(p["loss"], 4) for p in s], [10.0, 1.0, 20.0]))

    print(f"\n{sum(ok)}/{len(ok)} passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
