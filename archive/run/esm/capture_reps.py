"""
Run ESMFold2-Fast on a single complex (default 7txw) across many seeds and, for
each seed, save the intermediate representations:

  * ESMC language representation  (lm_hidden_states) — computed ONCE per seed,
    before the trunk loop, so it is saved once: reps/esmc_hidden_states.npy
  * Pairformer / folding-trunk representation z — updated every recycling
    iteration. The parcae recurrent trunk runs (num_loops + 1) times, so we save
    one tensor per iteration: reps/pairformer_z_loop{ii}.npy

Capture is non-invasive: a forward hook on model.folding_trunk fires once per
recycle (the confidence head uses a *separate* folding_trunk instance, so it is
not captured), and _compute_lm_hidden_states is wrapped to grab the ESMC rep.

Structures + confidence are saved exactly like run_foldbench_abag.py so the
seeds can still be scored with the normal eval pipeline.

Usage (one GPU, 20 seeds):
    CUDA_VISIBLE_DEVICES=0 ESM_PYTHON_run \
        python run/esm/capture_reps.py --seeds $(seq 0 19) --run_name 7txw_reps
"""

import argparse
import json
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


def load_complex(json_dir: str, name: str) -> StructurePredictionInput:
    """Load the base (non-updated) JSON for `name` and build its input."""
    json_dir = Path(json_dir)
    for json_path in sorted(json_dir.glob("*.json")):
        if "updated" in json_path.name or "update-msa" in json_path.name:
            continue
        with open(json_path) as f:
            data = json.load(f)
        entry = data[0]
        if entry["name"] != name:
            continue
        chain_idx = 0
        inputs = []
        for seq_entry in entry["sequences"]:
            if "proteinChain" not in seq_entry:
                continue
            chain_data = seq_entry["proteinChain"]
            count = chain_data.get("count", 1)
            ids = [CHAIN_LETTERS[chain_idx + i] for i in range(count)]
            chain_idx += count
            inputs.append(ProteinInput(id=ids if count > 1 else ids[0],
                                        sequence=chain_data["sequence"]))
        return StructurePredictionInput(sequences=inputs)
    raise SystemExit(f"Complex {name!r} not found in {json_dir}")


def _to_fp16_npy(t: torch.Tensor) -> np.ndarray:
    """Detach -> cpu -> float32 (avoids bf16 numpy gap) -> float16 for storage."""
    return t.detach().float().cpu().numpy().astype(np.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="7txw_reps")
    parser.add_argument("--complex", default="7txw")
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--lm_mask_pct", type=float, default=0.1)
    parser.add_argument("--lm_dropout", type=float, default=0.3)
    parser.add_argument("--json_dir", default=str(JSON_DIR))
    parser.add_argument("--num_loops", type=int, default=16)
    parser.add_argument("--num_sampling_steps", type=int, default=16)
    parser.add_argument("--num_diffusion_samples", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--no_config_log", action="store_true")
    args = parser.parse_args()

    name = args.complex
    print("Loading ESMFold2-Fast...", flush=True)
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(args.chunk_size if args.chunk_size > 0 else None)
    builder = ESMFold2InputBuilder()
    spi = load_complex(args.json_dir, name)

    # --- capture hooks --------------------------------------------------------
    # Per-recycle pairformer z: model.folding_trunk is called once per parcae
    # iteration (the confidence head has its own folding_trunk, not this one).
    trunk_reps: list[np.ndarray] = []

    def trunk_hook(_module, _inp, out):
        z = out[0] if isinstance(out, tuple) else out
        trunk_reps.append(_to_fp16_npy(z))

    model.folding_trunk.register_forward_hook(trunk_hook)

    # ESMC language rep: wrap the (single) compute call fold() makes per seed.
    esmc_box: dict[str, np.ndarray] = {}
    _orig_compute_lm = model._compute_lm_hidden_states

    def _wrapped_compute_lm(*a, **k):
        out = _orig_compute_lm(*a, **k)
        esmc_box["rep"] = _to_fp16_npy(out)
        return out

    model._compute_lm_hidden_states = _wrapped_compute_lm
    # --------------------------------------------------------------------------

    out_dir = ESM_PREDICTIONS / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_config_log:
        save_run_config(out_dir, {"model": "biohub/ESMFold2-Fast",
                                  "complex": name, **vars(args)})

    expected_trunk = args.num_loops + 1  # parcae runs num_loops + 1 times

    for seed_val in args.seeds:
        seed_int = int(seed_val)
        seed_dir = out_dir / name / f"seed_{seed_int}"
        pred_dir = seed_dir / "predictions"
        reps_dir = seed_dir / "reps"

        if (reps_dir / "esmc_hidden_states.npy").exists() and \
           any(pred_dir.glob(f"{name}_sample_*.cif")):
            print(f"  seed_{seed_int}  (skip — already exists)", flush=True)
            continue

        print(f"  seed_{seed_int} ...", end="", flush=True)
        trunk_reps.clear()
        esmc_box.clear()
        try:
            result = builder.fold(
                model, spi,
                seed=seed_int,
                num_loops=args.num_loops,
                num_sampling_steps=args.num_sampling_steps,
                num_diffusion_samples=args.num_diffusion_samples,
                lm_mask_pct=args.lm_mask_pct,
                lm_dropout=args.lm_dropout,
            )
            samples = result if isinstance(result, list) else [result]

            # --- save structures + confidence (parity with run_foldbench_abag) -
            pred_dir.mkdir(parents=True, exist_ok=True)
            for k, sample in enumerate(samples):
                with open(pred_dir / f"{name}_sample_{k}.cif", "w") as f:
                    f.write(sample.complex.to_mmcif())
                iptm = sample.iptm if sample.iptm is not None else (sample.ptm or 0.0)
                conf = {
                    "ranking_score": iptm,
                    "iptm": sample.iptm,
                    "ptm": sample.ptm,
                    "chain_pair_iptm": sample.pair_chains_iptm.tolist()
                    if sample.pair_chains_iptm is not None else None,
                }
                with open(pred_dir / f"{name}_summary_confidence_sample_{k}.json", "w") as f:
                    json.dump(conf, f)
                if sample.plddt is not None:
                    np.save(pred_dir / f"{name}_plddt_sample_{k}.npy",
                            sample.plddt.half().cpu().numpy())
                if sample.pae is not None:
                    np.save(pred_dir / f"{name}_pae_sample_{k}.npy",
                            sample.pae.half().cpu().numpy())

            # --- save representations -----------------------------------------
            reps_dir.mkdir(parents=True, exist_ok=True)
            if "rep" not in esmc_box:
                raise RuntimeError("ESMC representation was not captured")
            np.save(reps_dir / "esmc_hidden_states.npy", esmc_box["rep"])
            if len(trunk_reps) != expected_trunk:
                print(f"  WARN: captured {len(trunk_reps)} trunk reps, "
                      f"expected {expected_trunk}", end="")
            for i, z in enumerate(trunk_reps):
                np.save(reps_dir / f"pairformer_z_loop{i:02d}.npy", z)

            with open(reps_dir / "meta.json", "w") as f:
                json.dump({
                    "complex": name,
                    "seed": seed_int,
                    "lm_mask_pct": args.lm_mask_pct,
                    "lm_dropout": args.lm_dropout,
                    "num_loops": args.num_loops,
                    "num_trunk_iters": len(trunk_reps),
                    "esmc_shape": list(esmc_box["rep"].shape),
                    "pairformer_z_shape": list(trunk_reps[0].shape) if trunk_reps else None,
                    "dtype": "float16",
                }, f, indent=2)

            best_iptm = max(s.iptm or s.ptm or 0.0 for s in samples)
            print(f"  best ipTM={best_iptm:.3f}  "
                  f"(esmc {esmc_box['rep'].shape}, {len(trunk_reps)} trunk reps "
                  f"{trunk_reps[0].shape if trunk_reps else None})", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
