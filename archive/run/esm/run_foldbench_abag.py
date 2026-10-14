"""
Run ESMFold2-Fast on FoldBench AbAg across multiple seeds.

Defaults to 10% mask + 30% dropout (TTS settings). Pass --lm_mask_pct 0.0 --lm_dropout 0.0
for a no-masking baseline.

Usage:
    # Baseline (no masking, no dropout):
    CUDA_VISIBLE_DEVICES=3 python scripts/run_foldbench_abag.py \
        --run_name baseline --seeds baseline_seed

    # TTS (10% mask + 30% dropout):
    CUDA_VISIBLE_DEVICES=3 python scripts/run_foldbench_abag.py \
        --run_name esmfold2fast_tts --seeds 0 1 2 3 4 5 6 7 8 9 \
        --lm_mask_pct 0.1 --lm_dropout 0.3
"""

import argparse
import json
import os
import string
from pathlib import Path

import numpy as np
import torch

from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

# --- repo-root bootstrap so `from paths import ...` works from any cwd ---
import sys
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, JSON_DIR
from runlog import save_run_config

CHAIN_LETTERS = string.ascii_uppercase


def load_complexes(json_dir: str, glob_pattern: str = "*.json") -> list[tuple[str, StructurePredictionInput]]:
    """Load all base (non-updated) JSON files and return (name, input) pairs."""
    json_dir = Path(json_dir)
    json_files = sorted(
        p for p in json_dir.glob(glob_pattern)
        if "updated" not in p.name and "update-msa" not in p.name
    )

    complexes = []
    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)

        entry = data[0]
        name = entry["name"]

        chain_idx = 0
        inputs = []
        total_chains = sum(
            s["proteinChain"].get("count", 1)
            for s in entry["sequences"] if "proteinChain" in s
        )
        if total_chains > len(CHAIN_LETTERS):
            print(f"WARNING: {name} has {total_chains} chains, exceeds {len(CHAIN_LETTERS)}-chain limit — skipping")
            continue
        for seq_entry in entry["sequences"]:
            if "proteinChain" not in seq_entry:
                continue
            chain_data = seq_entry["proteinChain"]
            sequence = chain_data["sequence"]
            count = chain_data.get("count", 1)
            ids = [CHAIN_LETTERS[chain_idx + i] for i in range(count)]
            chain_idx += count
            inputs.append(ProteinInput(id=ids if count > 1 else ids[0], sequence=sequence))

        complexes.append((name, StructurePredictionInput(sequences=inputs)))

    return complexes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True,
                        help="Base output name. Each seed writes to foldbench_predictions/<run_name>_seed<N>/")
    parser.add_argument("--seeds", nargs="+", default=list(range(10)))
    parser.add_argument("--lm_mask_pct", type=float, default=0.1)
    parser.add_argument("--lm_dropout", type=float, default=0.3)
    parser.add_argument("--json_dir", default=str(JSON_DIR))
    parser.add_argument("--json_files", nargs="+", default=None,
                        help="Specific JSON files to process (overrides --json_dir).")
    parser.add_argument("--complexes", nargs="+", default=None,
                        help="Only process these complex names (e.g. 7txw 7ua2).")
    parser.add_argument("--exclude_complexes", nargs="+", default=None,
                        help="Skip these complex names (e.g. oversized targets that OOM "
                             "the confidence head). Applied after --complexes.")
    parser.add_argument("--num_loops", type=int, default=16)
    parser.add_argument("--num_sampling_steps", type=int, default=16)
    parser.add_argument("--num_diffusion_samples", type=int, default=5)
    parser.add_argument("--weight_perturb_sigma", type=float, default=0.0,
                        help="If >0, add Gaussian noise w += sigma*N(0,1) to every "
                             "float weight, regenerated deterministically per seed "
                             "(RandOpt-style). 0 = off (default).")
    parser.add_argument("--perturb_seed", type=int, default=None,
                        help="If set (requires --weight_perturb_sigma>0), fix the weight "
                             "perturbation to THIS seed and apply it ONCE, decoupled from the "
                             "fold seed(s). One perturbed model then folds every complex/seed "
                             "below. Unset = legacy behavior (perturbation reseeded per fold seed).")
    parser.add_argument("--chunk_size", type=int, default=32,
                        help="Token-axis chunk for L² ops (triangle/OPM/pair-transition) "
                             "in both the folding trunk and the confidence head. Lower = "
                             "less GPU memory, slightly slower; numerically identical. "
                             "Default 32 (model default is 64). Pass 0 to disable chunking.")
    parser.add_argument("--fused", action="store_true",
                        help="Enable the fused Triton trimul kernel on the O(L^3) "
                             "triangle-multiply op (bf16 fwd). Method-preserving speedup; "
                             "matches the steering path's --fused so wall-clock is "
                             "comparable between baseline and steering.")
    parser.add_argument("--no_config_log", action="store_true",
                        help="Skip writing a config snapshot (set by the parallel "
                             "launcher, which logs the run once itself).")
    args = parser.parse_args()

    print("Loading ESMFold2-Fast...")
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    # Cap the L² intermediates in both the trunk and the confidence head; the
    # confidence head batches all diffusion samples through a full trunk pass and
    # is the main OOM source on the largest AbAg complexes. None = chunking off.
    model.set_chunk_size(args.chunk_size if args.chunk_size > 0 else None)
    if args.fused:                          # Triton trimul kernel on the O(L^3) op; method-preserving speedup
        model.set_kernel_backend("fused")
    builder = ESMFold2InputBuilder()

    # RandOpt-style weight perturbation. The model is loaded once and reused across
    # every (complex, seed), so we snapshot the pretrained weights and, per seed,
    # restore from the snapshot then add seed-deterministic Gaussian noise. Restoring
    # from the clone (rather than subtracting the noise back) makes seed -> weights an
    # exact function with no float drift across the 50+ perturb cycles.
    base_params = None
    if args.weight_perturb_sigma > 0.0:
        # Snapshot on CPU: a full GPU copy of the 6.5B-param weights pushes the
        # confidence-head memory spike (it batches all diffusion samples through a
        # trunk pass) into OOM on the largest AbAg complexes. Tensor.copy_ handles
        # the CPU->GPU restore each seed; the noise is still generated on-GPU.
        base_params = {n: p.detach().to("cpu", copy=True)
                       for n, p in model.named_parameters()}
        print(f"Weight perturbation ON: sigma={args.weight_perturb_sigma} "
              f"({len(base_params)} parameter tensors snapshotted to CPU)")

    def apply_weight_perturbation(seed_int):
        """Restore pretrained weights, then add seed-deterministic Gaussian noise.

        Noise is generated on CPU (a single seeded Generator, advanced across params in
        fixed order) and moved to each parameter's current device. fold() offloads the
        ESM-C submodule (_esmc) to CPU between seeds, so after the first fold the model
        spans both devices — a single fixed-device generator would hit a device
        mismatch. CPU generation also makes seed -> perturbation independent of where
        each parameter currently lives, so it's bit-identical every run regardless of
        fold's memory juggling. Decoupled from the global RNG that fold's _seed_context
        reseeds for diffusion sampling.
        """
        gen = torch.Generator(device="cpu").manual_seed(int(seed_int))
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(base_params[n])                       # restore base (CPU -> p.device)
            for n, p in model.named_parameters():
                if not p.is_floating_point():
                    continue
                noise = torch.randn(p.shape, generator=gen)   # fp32 on CPU
                p.add_((args.weight_perturb_sigma * noise).to(device=p.device, dtype=p.dtype))

    # Decoupled mode: apply one fixed perturbation ONCE, reused across every fold below
    # (fold() doesn't alter weight values, only shuffles _esmc between devices, so the
    # perturbation persists). This is the RandOpt "one expert folds many complexes" path.
    if base_params is not None and args.perturb_seed is not None:
        apply_weight_perturbation(args.perturb_seed)
        print(f"Applied fixed weight perturbation once (perturb_seed={args.perturb_seed}); "
              f"reused across all complexes/seeds.")

    if args.json_files:
        complexes = []
        for jf in args.json_files:
            complexes.extend(load_complexes(str(Path(jf).parent), glob_pattern=Path(jf).name))
    else:
        complexes = load_complexes(args.json_dir)
    if args.complexes:
        complexes = [(n, s) for n, s in complexes if n in args.complexes]
    if args.exclude_complexes:
        excluded = set(args.exclude_complexes)
        complexes = [(n, s) for n, s in complexes if n not in excluded]
    print(f"Loaded {len(complexes)} complexes from {args.json_dir}")

    out_dir = ESM_PREDICTIONS / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Standalone runs log their own config; under the parallel launcher this is
    # suppressed (--no_config_log) so the launcher records the run exactly once.
    if not args.no_config_log:
        save_run_config(out_dir, {"model": "biohub/ESMFold2-Fast", **vars(args)})

    for i, (name, spi) in enumerate(complexes):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(complexes)}] {name}")
        print(f"{'='*60}")

        for seed_val in args.seeds:
            try:
                seed_int = int(seed_val)
                seed_dir_name = f"seed_{seed_int}"
            except (ValueError, TypeError):
                seed_int = 0
                seed_dir_name = str(seed_val)

            pred_dir = out_dir / name / seed_dir_name / "predictions"
            if pred_dir.exists() and any(pred_dir.glob(f"{name}_sample_*.cif")):
                print(f"  {seed_dir_name}  (skip — already exists)")
                continue

            print(f"  {seed_dir_name} ...", end="", flush=True)
            try:
                if args.weight_perturb_sigma > 0.0 and args.perturb_seed is None:
                    apply_weight_perturbation(seed_int)   # legacy: reseed per fold seed
                result = builder.fold(
                    model,
                    spi,
                    seed=seed_int,
                    num_loops=args.num_loops,
                    num_sampling_steps=args.num_sampling_steps,
                    num_diffusion_samples=args.num_diffusion_samples,
                    lm_mask_pct=args.lm_mask_pct,
                    lm_dropout=args.lm_dropout,
                )
                samples = result if isinstance(result, list) else [result]
                pred_dir.mkdir(parents=True, exist_ok=True)
                for k, sample in enumerate(samples):
                    with open(pred_dir / f"{name}_sample_{k}.cif", "w") as f:
                        f.write(sample.complex.to_mmcif())
                    iptm = sample.iptm if sample.iptm is not None else (sample.ptm or 0.0)
                    conf = {
                        "ranking_score": iptm,
                        "iptm": sample.iptm,
                        "ptm": sample.ptm,
                        "chain_pair_iptm": sample.pair_chains_iptm.tolist() if sample.pair_chains_iptm is not None else None,
                    }
                    with open(pred_dir / f"{name}_summary_confidence_sample_{k}.json", "w") as f:
                        json.dump(conf, f)
                    if sample.plddt is not None:
                        np.save(pred_dir / f"{name}_plddt_sample_{k}.npy",
                                sample.plddt.half().cpu().numpy())
                    if sample.pae is not None:
                        np.save(pred_dir / f"{name}_pae_sample_{k}.npy",
                                sample.pae.half().cpu().numpy())
                best_iptm = max(s.iptm or s.ptm or 0.0 for s in samples)
                print(f"  best ipTM={best_iptm:.3f}")
            except Exception as e:
                print(f"  ERROR: {e}")
            finally:
                torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
