#!/usr/bin/env python
"""THE PROPOSED REPO LAYOUT once the organized trainer replaces phase_k_train.py.

Line counts are MEASURED where the file exists and ESTIMATED from the source span where it does
not -- the figure says which, because a plan that hides what has not been written yet is how a
port gets signed off before it works.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DONE, TODO, GONE = "#2e7d32", "#ef6c00", "#9e9e9e"

TODAY = [
    ("phases/phase_k_train.py", "8,513 lines -- ONE function of 8,402", TODO),
    ("phases/  (80 .py)", "trainer, data prep, eval, model mods, tests", TODO),
    ("run_logging/  (105 .py)", "plotters, probes, one-off analyses", TODO),
    ("shell_scripts/  (55 .sh)", "one launcher per arm", TODO),
]

PROPOSED = [
    ("trainer/", "", None),
    ("    cli.py",          "866   argument surface",               DONE),
    ("    config.py",       "226   derived config + refusals",      DONE),
    ("    checkpoints.py",  "137   resume guard",                   DONE),
    ("    step_plan.py",    "239   epoch + cross-seed step build",  DONE),
    ("    bootstrap.py",    "~823  model, freeze, injections",      TODO),
    ("    unit.py",         "~721  run_unit: one pose end to end",  TODO),
    ("    loop.py",         "~977  train_loop",                     TODO),
    ("    evaluate.py",     "~690  held-out + val-loss passes",     TODO),
    ("    tests/",          "4 equivalence tests, passing",         DONE),
    ("model/", "", None),
    ("    pose_direction.py",   "orientation features",             DONE),
    ("    pose_perblock.py",    "sites 1-4",                        DONE),
    ("    pose_inblock.py",     "in-block variant",                 DONE),
    ("    pose_readout.py",     "site 5",                           DONE),
    ("    dead_ops.py",         "10 of 20 modules, measured",       DONE),
    ("    lean_readouts.py",    "readout trimming",                 DONE),
    ("    prefix_cache.py",     "frozen-prefix reuse",              DONE),
    ("    rank_loss.py",        "the ranking term",                 DONE),
    ("    lr_schedule.py",      "staircase + round-robin deal",     DONE),
    ("    pae_row_weight.py",   "PAE row weighting",                DONE),
    ("data/", "", None),
    ("    phase_a .. phase_j", "40 files, dataset build, unchanged", DONE),
    ("eval/", "", None),
    ("    phase_k_eval.py",         "the fold scorer",              DONE),
    ("    phase_k_eval_sharded.py", "sharded across cards",         DONE),
    ("    consistency checks",      "crop, perblock, fingerprint",  DONE),
    ("plotting/", "", None),
    ("    epoch_curve.py",      "one definition of a curve",        DONE),
    ("    eval_curve.py",       "held-out series + shipped",        DONE),
    ("    plot_rel_arm.py",     "THE training plot",                DONE),
    ("    plot_eval_series.py", "THE held-out plot",                DONE),
    ("    refresh_plots.sh",    "the only entry point",             DONE),
    ("abag_rank/", "", None),
    ("    11 .py", "the ab-ag ranking analysis line", DONE),
    ("analysis/", "", None),
    ("    3 .py", "one-off result analyses", DONE),
    ("scripts/", "", None),
    ("    launch_fullhead_5site.sh", "the live arm",                DONE),
    ("    lr_sweep_ep5.sh",         "the LR sweep",                 DONE),
    ("    mem_probe_fullhead.sh",   "the memory gate",              DONE),
]

REMOVED = [
    "phases/phase_k_train.py        the 8,402-line main()",
    "phases/lora.py                 LoRA: owner dropped it",
    "phases/itm_quality_loss.py     non-PAE / non-ranking losses",
    "50 CLI flags                   crop, L2-SP, early stop, decay",
    "~45 superseded launch scripts  one per retired arm",
    "one-off analyses in run_logging/",
]


def main(out="figures/repo_layout.png"):
    fig = plt.figure(figsize=(16.5, 12.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.35], width_ratios=[1, 1.32],
                          hspace=0.16, wspace=0.07)

    axT = fig.add_subplot(gs[0, 0]); axT.axis("off")
    axT.set_title("TODAY", fontsize=14, fontweight="bold", loc="left", color="#444")
    y = 0.90
    for name, note, col in TODAY:
        axT.text(0.02, y, name, fontsize=11.5, family="monospace", fontweight="bold")
        axT.text(0.05, y - 0.095, note, fontsize=10, color="#555")
        y -= 0.235

    axR = fig.add_subplot(gs[1, 0]); axR.axis("off")
    axR.set_title("REMOVED", fontsize=14, fontweight="bold", loc="left", color=GONE)
    y = 0.95
    for r in REMOVED:
        axR.text(0.02, y, r, fontsize=10, family="monospace", color=GONE)
        y -= 0.062

    axP = fig.add_subplot(gs[:, 1]); axP.axis("off")
    axP.set_title("PROPOSED  extra_training/", fontsize=14, fontweight="bold", loc="left")
    y = 0.975
    for name, note, col in PROPOSED:
        if col is None:                      # a package header
            y -= 0.008
            axP.add_patch(FancyBboxPatch((0.005, y - 0.019), 0.99, 0.030,
                                         boxstyle="round,pad=0.004",
                                         fc="#eceff1", ec="#b0bec5", lw=0.8,
                                         transform=axP.transAxes, zorder=0))
            axP.text(0.015, y - 0.004, name, fontsize=11.5, family="monospace",
                     fontweight="bold", va="center")
            y -= 0.0385
            continue
        axP.text(0.030, y, name, fontsize=9.8, family="monospace", va="center", color=col)
        axP.text(0.470, y, note, fontsize=9.0, va="center", color="#555")
        y -= 0.0194

    fig.suptitle("extra_training/ -- CODE layout after the trainer is reorganised",
                 fontsize=16.5, y=0.985)
    fig.text(0.5, 0.957,
             "SCOPE: the ~190 .py/.sh files that make up the training program and its tooling. "
             "NOT the whole repo.",
             ha="center", fontsize=10.5, color="#777")
    fig.text(0.5, 0.020,
             "green = written and CPU-tested     orange = still to port from phase_k_train.py's main()     "
             "grey = deleted\n"
             "~3,200 lines remain to port; sizes marked ~ are the source span, not a written file.\n"
             "OUT OF SCOPE, untouched: training/ validation/ (448k data files), building_msa/, "
             "SAbDab2_data/, docs/, figures/, train_val_splits/,\n"
             "and everything above extra_training/ (archive/ 647 .py, envs/ 13,103 .py, "
             "experimental_data/, tools/, run_model/).",
             ha="center", fontsize=10.5, color="#444")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
