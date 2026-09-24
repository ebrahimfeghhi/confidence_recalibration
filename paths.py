"""
Centralized paths for the tts_abag benchmark.

Single source of truth for every absolute path the run / eval / plot / analysis
scripts depend on. Each value defaults to a location under $HOME and can be
overridden with an environment variable of the same name, e.g.

    FOLDBENCH_DIR=/data/FoldBench python eval/eval_ost_foldbench.py ...

Scripts import this module via a small repo-root bootstrap (they walk up parent
dirs until they find this file, add the repo root to sys.path, then
`from paths import ...`). That keeps them runnable from any cwd and under any of
the model venvs without needing an editable install.

NOTE: this module points at *existing* data/output locations in the original
repos. The tts_abag repo holds scripts only — predictions and ground-truth data
are NOT moved here.
"""

import os
from pathlib import Path

HOME = Path.home()

# Repo root (directory containing this file) and default plot/CSV output dir.
REPO_ROOT = Path(__file__).resolve().parent


def _path(env_var: str, default: Path) -> Path:
    """Return Path from env override or default."""
    val = os.environ.get(env_var)
    return Path(val).expanduser() if val else default


# --- External input data (FoldBench ground truths + targets) ---------------
FOLDBENCH_DIR = _path("FOLDBENCH_DIR", HOME / "FoldBench")
GROUND_TRUTHS = _path("GROUND_TRUTHS", FOLDBENCH_DIR / "examples" / "ground_truths")
TARGETS_CSV = _path(
    "TARGETS_CSV", FOLDBENCH_DIR / "targets" / "interface_antibody_antigen.csv"
)

# Input JSONs (sequence/complex definitions) — currently live in the Protenix repo.
JSON_DIR = _path("JSON_DIR", HOME / "Protenix" / "foldbench_abag" / "jsons")

# --- Model repos / environments / checkpoints ------------------------------
# run_benchmark.sh must run from the Protenix repo (it calls runner/inference.py).
ESM_REPO = _path("ESM_REPO", HOME / "esm")
PROTENIX_REPO = _path("PROTENIX_REPO", HOME / "Protenix")
ESM_PYTHON = _path("ESM_PYTHON", ESM_REPO / ".venv" / "bin" / "python")
PROTENIX_PYTHON = _path("PROTENIX_PYTHON", PROTENIX_REPO / ".venv" / "bin" / "python")
PROTENIX_CKPT = _path("PROTENIX_CKPT", HOME / "checkpoint")

# --- OST (lives in the `foldbench` conda env) ------------------------------
OST_BIN = _path("OST_BIN", HOME / "miniconda3" / "envs" / "foldbench" / "bin" / "ost")
OST_LIB = _path("OST_LIB", HOME / "miniconda3" / "envs" / "foldbench" / "lib")

# --- Prediction outputs (data stays in the original repos) -----------------
ESM_PREDICTIONS = _path("ESM_PREDICTIONS", HOME / "esm" / "foldbench_predictions")
PROTENIX_PREDICTIONS = _path(
    "PROTENIX_PREDICTIONS",
    HOME / "Protenix" / "foldbench_abag" / "predictions_template",
)

# --- Plot / CSV outputs ----------------------------------------------------
RESULTS_DIR = _path("RESULTS_DIR", REPO_ROOT / "results")
# Eval CSVs (one row per sample) live in a dedicated subdir so plots/figures
# in RESULTS_DIR don't mingle with the raw data they're built from.
CSV_DIR = _path("CSV_DIR", RESULTS_DIR / "csv_files")

# --- ConforNet portable bundle ---------------------------------------------
# Self-contained ~300 MB bundle (centroids + baseline rep + 7txw json + ground
# truth + targets) produced by archive/confornet/prepare_data.py, so the steering +
# decode->DockQ experiment runs on a fresh server without the 195 GB 7txw_reps.
STEER_DATA = _path("STEER_DATA", REPO_ROOT / "steer" / "data")


# --- Current experiment: confident decoys / self-templating ------------------
# These moved out of confident_decoy_experiments/ when that era was archived: code now lives in
# diffusion_commit/ and everything the runs read or produce lives in experimental_data/. They are
# defined HERE rather than in diffusion_commit so there is one source of truth for locations, which
# is what made the move a one-line edit instead of a grep across a dozen files.
CODE_DIR = _path("CODE_DIR", REPO_ROOT / "diffusion_commit")
DATA_DIR = _path("DATA_DIR", REPO_ROOT / "experimental_data")
EVAL_DIR = _path("EVAL_DIR", REPO_ROOT / "eval")
# The pxm scorer is a built environment, not code we edit, and every DockQ score shells out to it --
# so it lives outside the archived repo.
PXMETER_BIN = _path("PXMETER_BIN", REPO_ROOT / "envs" / "pxmeter" / "bin" / "pxm")

# A pool travels as a unit: job jsons to fold, ground-truth cifs to score against, and the scorer
# that matches them. Mixing them silently produces nothing.
PXM2025_DIR = _path("PXM2025_DIR", DATA_DIR / "pxm2025")
PXM2025_JOBS = _path("PXM2025_JOBS", PXM2025_DIR / "jsons")
PXM2025_CIF = _path("PXM2025_CIF", PXM2025_DIR / "cif")
PXM2025_POOL_CSV = _path("PXM2025_POOL_CSV", PXM2025_DIR / "dockq_pxm2025.csv")
PXM2025_CHAIN_ROLES = _path("PXM2025_CHAIN_ROLES", PXM2025_DIR / "chain_roles.csv")


def dataset(name):
    """(job dir, ground-truth cif for a pdb, scorer name) for a pool, as strings."""
    if name == "pxm2025":
        return str(PXM2025_JOBS), (lambda pdb: f"{PXM2025_CIF}/{pdb}.cif"), "pxm"
    if name == "foldbench":
        return str(JSON_DIR), (lambda pdb: f"{GROUND_TRUTHS}/{pdb}-assembly1.cif"), "ost"
    raise KeyError(f"unknown dataset {name!r}")


DATASET_NAMES = ("pxm2025", "foldbench")


# --- run trees ---------------------------------------------------------------
# Every fold we generate lands in one of these two, as <pdb>/<seed>/<run_id>/.
FREE_DIR = _path("FREE_DIR", DATA_DIR / "free")
# anchor_relabel.py writes the chain maps, roles and manifest here; pose_interfaces.py and the queue
# builders read them. Points at the existing data rather than a new location so nothing has to move.
ANCHOR_DIR = _path("ANCHOR_DIR", DATA_DIR / "anchor_frame")
# code locations, so scripts do not carry literals for each other's directories
METRICS_DIR = _path("METRICS_DIR", REPO_ROOT / "diffusion_commit_clean" / "metrics")
CLEAN_CODE_DIR = _path("CLEAN_CODE_DIR", REPO_ROOT / "diffusion_commit_clean")
EVAL_DIR = _path("EVAL_DIR", REPO_ROOT / "eval")
PROTENIX_SRC = _path("PROTENIX_SRC", Path.home() / "Protenix")
SELFTMPL_DIR = _path("SELFTMPL_DIR", DATA_DIR / "self_template")
# The template namespace write_templates.py builds: an input to self-templated folds.
TEMPLATES_DIR = _path("TEMPLATES_DIR", SELFTMPL_DIR / "templates")

# The pool's original 50-75 seeds per complex predate run provenance -- no run id, no run.json, and
# their configuration is known only from what the defaults were at the time (bf16, N_cycle 10). They
# sit under this marker so the absence is visible in the path rather than dressed up as a real run id.
LEGACY_RUN = "legacy-no-provenance"


def pool_pose(pdb, seed, sample):
    """Path to one pose of the original pool. A FUNCTION, not a format string: legacy folds and
    modern ones nest predictions differently, and two scripts had hand-built this path identically,
    which is how a layout change breaks them both silently.

    It then broke ANYWAY: the run-id migration flattened <seed>/<run_id>/ to <seed>-<run_id>/ and
    this function kept returning the old nested form, so it pointed at nothing for months. Nobody
    noticed because both callers glob or guard. Centralising a path is necessary but not sufficient:
    nothing here fails loudly when the layout moves, so a layout change needs this checked by hand."""
    return (f"{FREE_DIR}/{pdb}/{seed}-{LEGACY_RUN}/predictions/{pdb}_sample_{sample}.cif")


def _sh():
    """Print `export NAME=value` lines so shell scripts can source these paths:

        eval "$(python paths.py --sh)"
    """
    for name in (
        "FOLDBENCH_DIR", "GROUND_TRUTHS", "TARGETS_CSV", "JSON_DIR",
        "ESM_REPO", "PROTENIX_REPO", "ESM_PYTHON", "PROTENIX_PYTHON",
        "PROTENIX_CKPT", "OST_BIN", "OST_LIB",
        "ESM_PREDICTIONS", "PROTENIX_PREDICTIONS", "RESULTS_DIR", "CSV_DIR",
        "STEER_DATA", "CODE_DIR", "DATA_DIR", "EVAL_DIR", "PXMETER_BIN",
        "PXM2025_DIR", "PXM2025_JOBS", "PXM2025_CIF", "PXM2025_POOL_CSV",
        "FREE_DIR", "SELFTMPL_DIR", "TEMPLATES_DIR",
    ):
        print(f'export {name}="{globals()[name]}"')


if __name__ == "__main__":
    import sys

    if "--sh" in sys.argv:
        _sh()
    else:
        for name, val in sorted(globals().items()):
            if name.isupper() and isinstance(val, Path):
                exists = "ok " if val.exists() else "MISSING"
                print(f"[{exists}] {name} = {val}")
