"""
Centralized paths for the confidence_recalibration benchmark (formerly tts_abag).

Single source of truth for every absolute path the run / eval / plot / analysis
scripts depend on. Each value defaults to a location under $HOME and can be
overridden with an environment variable of the same name, e.g.

    FOLDBENCH_DIR=/data/FoldBench python archive/eval_ost_foldbench.py ...

Scripts import this module via a small repo-root bootstrap (they walk up parent
dirs until they find this file, add the repo root to sys.path, then
`from paths import ...`). That keeps them runnable from any cwd and under any of
the model venvs without needing an editable install.

NOTE: this module points at *existing* data/output locations in the original
repos. This repo holds scripts only — predictions and ground-truth data
are NOT moved here.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

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
# pxmeter's chain permutation lives in its own venv: it is the only dependency the Protenix venv lacks
PXMETER_PYTHON = _path("PXMETER_PYTHON", REPO_ROOT / "envs" / "pxmeter" / "bin" / "python")
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
CODE_DIR = _path("CODE_DIR", REPO_ROOT / "archive" / "diffusion_commit")
DATA_DIR = _path("DATA_DIR", REPO_ROOT / "experimental_data")
EVAL_DIR = _path("EVAL_DIR", REPO_ROOT / "eval")
# dockq_torch lives here; the GPU pose-to-pose DockQ imports it
CLUSTERING_DIR = _path("CLUSTERING_DIR", REPO_ROOT / "archive" / "diffusion_commit" / "clustering")
# pose-to-pose DockQ matrices: ONE ROOT, one subdirectory per (dataset, frame). See its README.md.
# A dataset's crystal-frame set is complete -- every interface of the pool has an npz -- so these are
# read, never rebuilt. Dataset.matrices points at the right one for that dataset.
POSE_DOCKQ_MATRICES = _path("POSE_DOCKQ_MATRICES", REPO_ROOT / "archive" / "pose_dockq_matrices")
# confidence-head finetune project root; everything for it lives here
EXTRA_TRAINING = _path("EXTRA_TRAINING", REPO_ROOT / "extra_training")
# The three names that used to live here -- XTAL_MATRICES, ANCHOR_MATRICES, SPUR_MATRICES -- pointed at
# archive/superseded_2026-09-25 directories that are EMPTY, and nothing read them. They are gone rather
# than left as live-looking constants resolving to nothing: a caller that globbed one got zero files and
# no error. Use POSE_DOCKQ_MATRICES / Dataset.matrices.
# The pxm scorer is a built environment, not code we edit, and every DockQ score shells out to it --
# so it lives outside the archived repo.
PXMETER_BIN = _path("PXMETER_BIN", REPO_ROOT / "envs" / "pxmeter" / "bin" / "pxm")

# A pool travels as a unit: job jsons to fold, ground-truth cifs to score against, and the scorer
# that matches them. Mixing them silently produces nothing.
PXM2025_DATA = _path("PXM2025_DATA", DATA_DIR / "protenix" / "pxm2025")
# pxm2025 now mirrors foldbench: inputs AND arms under one benchmark directory.
# Before 2026-10-06 its data root WAS experimental_data itself, which is why two
# different pools were both called free/ at different depths.
PXM2025_DIR = _path("PXM2025_DIR", PXM2025_DATA)
PXM2025_JOBS = _path("PXM2025_JOBS", PXM2025_DIR / "jsons")
PXM2025_CIF = _path("PXM2025_CIF", PXM2025_DIR / "cif")
# THE INTERFACE REGISTRY, not a results table -- folder.py reads only pdb/chain1/chain2 from it.
# dockq_pxm2025.csv used to carry that list as a side effect of being the scores, which is why the
# scores could not be deleted. pxm2025_interfaces.csv is the 292 folded interfaces, one row each,
# derived from pxm2025_baseline_results.csv. chain_roles.csv and interfaces.csv both list 314 and so
# are NOT substitutes: 22 of theirs were never folded.
PXM2025_POOL_CSV = _path("PXM2025_POOL_CSV", PXM2025_DIR / "pxm2025_interfaces.csv")
PXM2025_CHAIN_ROLES = _path("PXM2025_CHAIN_ROLES", PXM2025_DIR / "chain_roles.csv")


# --- run trees ---------------------------------------------------------------
# Every fold we generate lands in one of these two, as <pdb>/<seed>/<run_id>/.
FREE_DIR = _path("FREE_DIR", PXM2025_DATA / "free")
# one directory for every benchmark's free seeds, assembled per benchmark into a single
# pose-level table. Separate from each benchmark's own experiment tree because the whole point is
# that an analysis should not have to know a benchmark lives under protenix/ or beside it.
# NOT a third top-level directory. Each benchmark's pooled free table lives inside that
# benchmark's own folder as `free_pool/`, so there is exactly one directory per benchmark and no
# analysis has to know whether a benchmark sits under protenix/ or beside it. An earlier
# experimental_data/free/ held the pxm pose pool, was migrated away, and is still referenced by
# stale docstrings -- reusing that name would have revived the ambiguity.

# FoldBench, run the same way. Its experiments live under experimental_data/protenix/foldbench/; its
# inputs there are built once by archive/diffusion_commit_clean_v2/foldbench_setup.py from the
# FoldBench 50-seed Protenix run, whose poses were deleted 2026-10-14; its scores are in
# foldbench_baseline_results.csv.
FOLDBENCH_DATA = _path("FOLDBENCH_DATA", DATA_DIR / "protenix" / "foldbench")


@dataclass(frozen=True)
class Dataset:
    """Everything a pipeline step needs to know about one benchmark, so no step hardcodes one.

    data:        where its experiments live (free_compute_matched/, self_template_*/ ...)
    free:        the free pool; every template's donor is a pose under here
    jobs:        Protenix job jsons, <pdb>-final-updated.json
    ref:         pdb -> the ground-truth cif (EVALUATION ONLY)
    scorer:      "pxm" or "ost", matching how the pool was scored
    pool_csv:    the crystal-scored pool: pdb, chain1, chain2, seed, sample, dockq, pair_iptm (EVALUATION,
                 plus which complexes form the benchmark)
    chain_maps:  crystal chain map per pose (reference chain -> pose chain); used for ROLES only
    chain_roles: the benchmark's antibody / antigen chains (interface, pdb, chain_1, chain_2, ab_chain,
                 ag_chain)
    matrices:    pose-to-pose DockQ matrices in the CRYSTAL frame, keyed by interface name exactly as
                 chain_roles names it, so `matrices / f"{iface}.npz"` resolves. Complete for the pool
    pool_seeds:  the seeds the pool rule admits, in order; the first 50 define the free baseline
    free_pool:   EVERY free pose-level score for this benchmark in one table, pool_csv plus the
                 later top-up arms, with a `block` column saying which arm each row came from.
                 Built by diffusion_commit_clean_v2/build_free_pool.py; INDEX.md beside it records
                 where each block's poses are. Deeper than pool_csv (foldbench 60 seeds, pxm2025 a
                 median of 70), so prefer it for anything seed-depth dependent and use pool_csv
                 only when the free-50 baseline itself is the point
    free_templates: whether a FREE fold uses Protenix's PDB templates. The free folds that make up a
                 dataset's pool decide it, and every later free fold must match them: pxm2025's used 3-4
                 per chain; FoldBench's used none (FoldBench's own Protenix setup disables templates,
                 configs_data.py "template": {"enable": False}), so its free folds run template-free
    """
    name: str
    data: Path
    free: Path
    jobs: Path
    ref: Callable[[str], str]
    scorer: str
    pool_csv: Path
    chain_maps: Path
    chain_roles: Path
    matrices: Path
    pool_seeds: Tuple[str, ...]
    free_templates: bool = True
    free_pool: Optional[Path] = None


DATASETS = {
    "pxm2025": Dataset(
        name="pxm2025", data=PXM2025_DATA, free=FREE_DIR, jobs=PXM2025_JOBS,
        ref=lambda pdb: f"{PXM2025_CIF}/{pdb}.cif", scorer="pxm", pool_csv=PXM2025_POOL_CSV,
        chain_maps=PXM2025_DIR / "chain_maps.csv", chain_roles=PXM2025_CHAIN_ROLES,
        matrices=POSE_DOCKQ_MATRICES / "pxm2025_crystal",
        pool_seeds=tuple(str(300000 + i) for i in range(50)),
        free_pool=PXM2025_DATA / "free_pool" / "results.csv"),
    "foldbench": Dataset(
        name="foldbench", data=FOLDBENCH_DATA, free=FOLDBENCH_DATA / "free", jobs=JSON_DIR,
        ref=lambda pdb: f"{GROUND_TRUTHS}/{pdb}-assembly1.cif", scorer="ost",
        # THE INTERFACE REGISTRY, not a results table. folder.py reads only pdb/chain1/chain2 from
        # this file; FoldBench's 50-seed scores used to carry that list as a side effect, which is
        # why they could not be deleted. chain_roles.csv is one row per interface -- 160 against
        # 39,998 -- and says what it is. The scores now live in foldbench_baseline_results.csv.
        pool_csv=FOLDBENCH_DATA / "chain_roles.csv", chain_maps=FOLDBENCH_DATA / "chain_maps.csv",
        chain_roles=FOLDBENCH_DATA / "chain_roles.csv",
        matrices=POSE_DOCKQ_MATRICES / "foldbench_crystal",
        pool_seeds=tuple(str(101 + i) for i in range(50)), free_templates=False,
        free_pool=FOLDBENCH_DATA / "foldbench_baseline_results.csv"),
    # the 483 training clusters, one representative each. Same pipeline as sabdab_val and a
    # DISJOINT cluster set -- these are the poses the confidence head is finetuned ON, so nothing
    # here may share a store with the validation fold it is evaluated against.
    "sabdab_train": Dataset(
        name="sabdab_train", data=EXTRA_TRAINING, free=EXTRA_TRAINING / "training" / "fold",
        jobs=EXTRA_TRAINING / "training" / "jobs",
        ref=lambda pdb: f"{EXTRA_TRAINING}/training/refs/{pdb}.cif", scorer="ost",
        pool_csv=EXTRA_TRAINING / "training" / "fold" / "pool.csv",
        chain_maps=EXTRA_TRAINING / "training" / "fold" / "chain_maps.csv",
        chain_roles=EXTRA_TRAINING / "training" / "fold" / "chain_roles.csv",
        matrices=POSE_DOCKQ_MATRICES / "sabdab_train_crystal",
        pool_seeds=tuple(str(500000 + i) for i in range(20))),
    # The REDESIGNED validation split: 40 post-cutoff clusters, one named representative each,
    # chosen by scripts/propose_val_split.py and reviewed as extra_training/docs/val_split_proposal.md.
    # Separate from sabdab_val because the two coexist during the changeover -- sabdab_val's fold is
    # what the 21-cluster set was measured on, and overwriting its pool.csv would make those numbers
    # unreproducible.
    "sabdab_val2": Dataset(
        name="sabdab_val2", data=EXTRA_TRAINING, free=EXTRA_TRAINING / "validation" / "fold",
        jobs=EXTRA_TRAINING / "validation" / "jobs",
        ref=lambda pdb: f"{EXTRA_TRAINING}/validation/refs/{pdb}.cif", scorer="ost",
        pool_csv=EXTRA_TRAINING / "validation" / "fold" / "pool.csv",
        chain_maps=EXTRA_TRAINING / "validation" / "fold" / "chain_maps.csv",
        chain_roles=EXTRA_TRAINING / "validation" / "fold" / "chain_roles.csv",
        matrices=POSE_DOCKQ_MATRICES / "sabdab_val2_crystal",
        pool_seeds=tuple(str(400000 + i) for i in range(100))),
}
DATASET_NAMES = tuple(DATASETS)


def dataset(name):
    """(job dir, ground-truth cif for a pdb, scorer name) for a pool, as strings -- the older
    three-field view of DATASETS[name]."""
    d = DATASETS[name]
    return str(d.jobs), d.ref, d.scorer
# anchor_relabel.py writes the chain maps, roles and manifest here; pose_interfaces.py and the queue
# builders read them. Points at the existing data rather than a new location so nothing has to move.
ANCHOR_DIR = _path("ANCHOR_DIR", DATA_DIR / "archive" / "superseded_2026-09-25" / "anchor_frame")
# code locations, so scripts do not carry literals for each other's directories
METRICS_DIR = _path("METRICS_DIR", REPO_ROOT / "archive" / "diffusion_commit_clean" / "metrics")
CLEAN_CODE_DIR = _path("CLEAN_CODE_DIR", REPO_ROOT / "archive" / "diffusion_commit_clean")
PROTENIX_SRC = _path("PROTENIX_SRC", Path.home() / "Protenix")

# The pool's original 50-75 seeds per complex predate run provenance -- no run id, no run.json, and
# their configuration is known only from what the defaults were at the time (bf16, N_cycle 10). They
# sit under this marker so the absence is visible in the path rather than dressed up as a real run id.
LEGACY_RUN = "legacy-no-provenance"


def pool_pose(pdb, seed, sample, free_dir=None):
    """Path to one pose of a free pool (default FREE_DIR, pxm2025's), whatever run id its fold was
    written under.

    The legacy path is tried FIRST and returned if it exists, so every pose that resolved before this
    function learned to glob still resolves to exactly the same file. Only seeds outside the legacy
    naming reach the glob -- before, they were unreachable, which is why 4,000 folds sat on disk
    invisible to selection.

    Raises on a seed with two fold directories holding poses. Folds do not reproduce, so those are
    different data and picking one would be a silent choice; there are none today.
    """
    free_dir = free_dir or FREE_DIR
    legacy = f"{free_dir}/{pdb}/{seed}-{LEGACY_RUN}/predictions/{pdb}_sample_{sample}.cif"
    if os.path.exists(legacy):
        return legacy
    # search the whole fold directory: the legacy layout is <seed>-<rid>/predictions/<pdb>_sample_N.cif
    # while the current driver writes <seed>-<rid>/<jobname>/seed_<seed>/predictions/<jobname>_sample_N.cif
    # -- deeper, and the filename carries the job name, so both the depth and the prefix differ
    import glob as _glob
    hits = sorted(_glob.glob(f"{free_dir}/{pdb}/{seed}-*/**/{pdb}*_sample_{sample}.cif",
                             recursive=True))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(f"{pdb} seed {seed} sample {sample}: {len(hits)} fold directories hold "
                           f"this pose, so which one is meant is ambiguous: {hits}")
    return legacy          # nothing found: return the legacy path, as before, for callers that guard


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
        "PXM2025_DATA", "PXM2025_DIR", "PXM2025_JOBS", "PXM2025_CIF", "PXM2025_POOL_CSV",
        "FREE_DIR", "POSE_DOCKQ_MATRICES",
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
