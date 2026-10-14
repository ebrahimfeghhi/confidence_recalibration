"""
Launch Protenix-v2 inference in parallel across GPUs, splitting SEEDS across GPUs.

Each GPU runs ALL structures for its subset of seeds (the same "seeds axis" as the
ESM launcher run_foldbench_abag_parallel.py). Structures vary ~250x in compute, so
splitting by seeds keeps every GPU on the same structure workload — no single GPU
gets stuck with all the large complexes.

Speed: each GPU makes a SINGLE inference.py call over a *merged* multi-structure
JSON, so the checkpoint loads once per GPU instead of once per structure
(infer_predict already sweeps the whole dataloader x seeds). Resume-aware:
structures already complete for a GPU's seeds are dropped before the call.

Usage (needs the Protenix venv + checkpoint):
    ~/Protenix/.venv/bin/python run/protenix/run_benchmark_parallel.py \
        --gpus 0 1 3 4 --seeds $(seq 101 110)

    # preview the commands without launching anything:
    ~/Protenix/.venv/bin/python run/protenix/run_benchmark_parallel.py \
        --gpus 0 1 3 4 --dry_run
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# --- repo-root bootstrap so `from paths import ...` works from any cwd ---
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import (
    PROTENIX_REPO, PROTENIX_PYTHON, PROTENIX_CKPT, PROTENIX_PREDICTIONS, JSON_DIR,
)
from runlog import save_run_config

GLOB = "*-final-updated.json"
INFERENCE = "runner/inference.py"  # relative to PROTENIX_REPO (must be cwd)


def discover_structures(json_dir, complexes):
    """Return [(name, json_path)] for every input structure (optionally filtered)."""
    out = []
    for jp in sorted(Path(json_dir).glob(GLOB)):
        name = jp.name.replace("-final-updated.json", "")
        if complexes and name not in complexes:
            continue
        out.append((name, jp))
    return out


def seed_done(dump_dir, name, seed, n_sample):
    """True if this (structure, seed) already produced its last diffusion sample."""
    marker = (Path(dump_dir) / name / f"seed_{seed}" / "predictions"
              / f"{name}_summary_confidence_sample_{n_sample - 1}.json")
    return marker.exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(101, 111)))
    parser.add_argument("--dump_dir", default=str(PROTENIX_PREDICTIONS))
    parser.add_argument("--checkpoint", default=str(PROTENIX_CKPT))
    parser.add_argument("--json_dir", default=str(JSON_DIR))
    parser.add_argument("--model_name", default="protenix-v2")
    parser.add_argument("--n_sample", type=int, default=5)
    parser.add_argument("--n_step", type=int, default=200)
    parser.add_argument("--use_template", default="true")
    parser.add_argument("--complexes", nargs="+", default=None,
                        help="Only process these structure names (e.g. 7st5 7txw).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would run (per-GPU seeds, structure counts, "
                             "commands) without writing JSONs or launching anything.")
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)

    structures = discover_structures(args.json_dir, args.complexes)
    if not structures:
        print(f"No structures matching {GLOB} in {args.json_dir}")
        return
    print(f"Structures: {len(structures)}   Seeds: {args.seeds}   GPUs: {args.gpus}")

    n_gpus = len(args.gpus)
    seed_chunks = [args.seeds[i::n_gpus] for i in range(n_gpus)]

    if not args.dry_run:
        dump_dir.mkdir(parents=True, exist_ok=True)
        save_run_config(dump_dir, {"model": args.model_name, **vars(args)})

    procs = []
    for gpu, seeds in zip(args.gpus, seed_chunks):
        if not seeds:
            continue

        # Resume: keep a structure only if at least one of this GPU's seeds is missing.
        # (Included structures re-run all of this GPU's seeds; the runner overwrites
        # already-done seeds idempotently, so restart cost stays bounded.)
        todo = [(name, jp) for name, jp in structures
                if not all(seed_done(dump_dir, name, s, args.n_sample) for s in seeds)]
        if not todo:
            print(f"GPU {gpu}: seeds {seeds} — all structures already complete, skipping")
            continue

        seeds_csv = ",".join(str(s) for s in seeds)
        combined_path = dump_dir / f"_tts_combined_gpu{gpu}.json"
        cmd = [
            str(PROTENIX_PYTHON), INFERENCE,
            "--model_name", args.model_name,
            "--load_checkpoint_dir", str(args.checkpoint),
            "--input_json_path", str(combined_path),
            "--dump_dir", str(dump_dir),
            "--seeds", seeds_csv,
            "--sample_diffusion.N_sample", str(args.n_sample),
            "--sample_diffusion.N_step", str(args.n_step),
            "--use_template", args.use_template,
        ]

        if args.dry_run:
            merged = sum(len(json.loads(jp.read_text())) for _, jp in todo)
            print(f"\nGPU {gpu}: seeds {seeds}  |  {len(todo)} structures "
                  f"({merged} merged entries) -> {combined_path}")
            print("  CUDA_VISIBLE_DEVICES={} LAYERNORM_TYPE=openfold \\\n  {}"
                  .format(gpu, " ".join(cmd)))
            continue

        # Merge this GPU's structures into one JSON (one checkpoint load processes all).
        merged = []
        for name, jp in todo:
            merged.extend(json.loads(jp.read_text()))
        combined_path.write_text(json.dumps(merged))

        log_path = dump_dir / f"gpu{gpu}.log"
        log_file = open(log_path, "w")
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "LAYERNORM_TYPE": "openfold",
            # One long-lived process now sees structures spanning ~200-2700 tokens;
            # expandable segments reduces allocator fragmentation across those sizes.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        proc = subprocess.Popen(cmd, cwd=str(PROTENIX_REPO), env=env,
                                stdout=log_file, stderr=log_file)
        print(f"GPU {gpu}: seeds {seeds}  {len(todo)} structures  "
              f"(pid {proc.pid})  log → {log_path}")
        procs.append((proc, log_file))

    for proc, log_file in procs:
        proc.wait()
        log_file.close()

    if not args.dry_run and procs:
        print("\nAll done.")


if __name__ == "__main__":
    main()
