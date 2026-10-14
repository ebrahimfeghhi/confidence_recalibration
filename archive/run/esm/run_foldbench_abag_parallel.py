"""
Launch run_foldbench_abag.py in parallel across multiple GPUs, splitting seeds evenly.
Each GPU runs all structures for its subset of seeds.

Usage:
    python scripts/run_foldbench_abag_parallel.py \
        --run_name esmfold2fast_tts \
        --gpus 0 1 2 3 \
        --seeds 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# --- repo-root bootstrap so `from paths import ...` works from any cwd ---
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, ESM_PYTHON, JSON_DIR
from runlog import save_run_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", default=list(range(10)))
    parser.add_argument("--lm_mask_pct", type=float, default=0.1)
    parser.add_argument("--lm_dropout", type=float, default=0.3)
    parser.add_argument("--num_loops", type=int, default=16)
    parser.add_argument("--num_sampling_steps", type=int, default=16)
    parser.add_argument("--num_diffusion_samples", type=int, default=5)
    parser.add_argument("--weight_perturb_sigma", type=float, default=0.0,
                        help="If >0, RandOpt-style per-seed Gaussian weight noise "
                             "(forwarded to each worker). 0 = off (default).")
    parser.add_argument("--chunk_size", type=int, default=32,
                        help="Token-axis chunk for L² ops (forwarded to each worker). "
                             "Lower = less GPU memory, slightly slower; numerically "
                             "identical. Default 32. Pass 0 to disable chunking.")
    parser.add_argument("--fused", action="store_true",
                        help="Enable the fused Triton trimul kernel (forwarded to each "
                             "worker). Method-preserving speedup; matches steering's --fused.")
    parser.add_argument("--json_dir", default=str(JSON_DIR))
    parser.add_argument("--complexes", nargs="+", default=None,
                        help="Only process these complex names (e.g. 7txw 7ua2).")
    parser.add_argument("--exclude_complexes", nargs="+", default=None,
                        help="Skip these complex names (forwarded to each worker).")
    args = parser.parse_args()

    n_gpus = len(args.gpus)
    seed_chunks = [args.seeds[i::n_gpus] for i in range(n_gpus)]

    out_dir = ESM_PREDICTIONS / args.run_name
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Log the whole run once here; inner processes get --no_config_log so they
    # don't each write a (racy) duplicate.
    save_run_config(out_dir, {"model": "biohub/ESMFold2-Fast", **vars(args)})

    script = Path(__file__).resolve().parent / "run_foldbench_abag.py"
    procs = []
    for gpu, seeds in zip(args.gpus, seed_chunks):
        if not seeds:
            continue
        cmd = [
            str(ESM_PYTHON), str(script),
            "--run_name", args.run_name,
            "--seeds", *map(str, seeds),
            "--num_loops", str(args.num_loops),
            "--num_sampling_steps", str(args.num_sampling_steps),
            "--num_diffusion_samples", str(args.num_diffusion_samples),
            "--weight_perturb_sigma", str(args.weight_perturb_sigma),
            "--chunk_size", str(args.chunk_size),
            "--lm_mask_pct", str(args.lm_mask_pct),
            "--lm_dropout", str(args.lm_dropout),
            "--json_dir", args.json_dir,
            "--no_config_log",
        ]
        if args.fused:
            cmd += ["--fused"]
        if args.complexes:
            cmd += ["--complexes", *args.complexes]
        if args.exclude_complexes:
            cmd += ["--exclude_complexes", *args.exclude_complexes]
        log_path = log_dir / f"gpu{gpu}.log"
        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                                          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                                stdout=log_file, stderr=log_file)
        print(f"GPU {gpu}: seeds {seeds}  (pid {proc.pid})  log → {log_path}")
        procs.append((proc, log_file))

    for proc, log_file in procs:
        proc.wait()
        log_file.close()

    print("\nAll done.")


if __name__ == "__main__":
    main()
