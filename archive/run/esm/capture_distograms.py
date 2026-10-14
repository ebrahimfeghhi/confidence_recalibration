"""Capture the PER-LOOP cross-block (antibody x antigen) distogram for 8im0 across many
seeds, with TTS mask/dropout (matching the scale run). The decoded CIFs only give the
FINAL loop; this records the full recycling trajectory so we can see WHEN/WHERE the
interface forms. Saves the compact cross-block distogram (not the full z) to stay under
the disk quota.

Per seed -> ESM_PREDICTIONS/<run_name>/8im0/seed_<s>/cross_distogram_loops.npy
            shape [num_loops+1, |AB|*|C|, 64] float16  (AB=antibody rows, C=antigen cols)
          + confidence.json (best/per-sample iptm).

Usage (one GPU, 30 seeds):
  CUDA_VISIBLE_DEVICES=N python run/esm/capture_distograms.py \
      --run_name 8im0_distotraj --seeds $(seq 2000 2029)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from esm.models.esmfold2 import ESMFold2InputBuilder
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
sys.path.insert(0, str(_p / "steer" / "core"))
sys.path.insert(0, str(_p / "run" / "esm"))
from paths import ESM_PREDICTIONS, JSON_DIR          # noqa: E402
from runlog import save_run_config                    # noqa: E402
from complex_io import load_complex, build_conditioning  # noqa: E402
import forward_grad as fg                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="8im0_distotraj")
    ap.add_argument("--complex", default="8im0")
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--lm_mask_pct", type=float, default=0.1)   # match the scale run
    ap.add_argument("--lm_dropout", type=float, default=0.3)
    ap.add_argument("--num_loops", type=int, default=16)
    ap.add_argument("--num_sampling_steps", type=int, default=16)
    ap.add_argument("--num_diffusion_samples", type=int, default=2)
    ap.add_argument("--chunk_size", type=int, default=32)
    args = ap.parse_args()

    name = args.complex
    fg.configure_target(name)                          # sets fg.AB / fg.C (antibody / antigen slices)
    print(f"Loading ESMFold2-Fast... AB={fg.AB} C={fg.C}", flush=True)
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(args.chunk_size if args.chunk_size > 0 else None)
    builder = ESMFold2InputBuilder()
    spi = load_complex(str(JSON_DIR), name)

    # pair_mask is seed-independent (structural) -> compute once
    feat0, _ = builder.prepare_input(spi, seed=0, device=model.device)
    with torch.no_grad():
        pair_mask = build_conditioning(model, feat0)["pair_mask"]

    # capture each recycle's folding-trunk output z (kept on GPU, fp16, to convert later)
    zs: list[torch.Tensor] = []

    def trunk_hook(_m, _i, out):
        z = out[0] if isinstance(out, tuple) else out
        zs.append(z.detach().half())

    model.folding_trunk.register_forward_hook(trunk_hook)

    out_dir = ESM_PREDICTIONS / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(out_dir, {"model": "biohub/ESMFold2-Fast", "complex": name, **vars(args)})
    expected = args.num_loops + 1

    for sv in args.seeds:
        s = int(sv)
        sdir = out_dir / name / f"seed_{s}"
        if (sdir / "cross_distogram_loops.npy").exists():
            print(f"  seed_{s}  (skip)", flush=True)
            continue
        print(f"  seed_{s} ...", end="", flush=True)
        zs.clear()
        try:
            result = builder.fold(model, spi, seed=s,
                                  num_loops=args.num_loops,
                                  num_sampling_steps=args.num_sampling_steps,
                                  num_diffusion_samples=args.num_diffusion_samples,
                                  lm_mask_pct=args.lm_mask_pct, lm_dropout=args.lm_dropout)
            samples = result if isinstance(result, list) else [result]

            # per-loop cross-block distogram from each captured z
            dist = []
            with torch.no_grad():
                for z in zs:
                    P = fg.cross_distogram_from_z(model, z.float(), pair_mask)   # [npairs,64]
                    dist.append(P.half().cpu().numpy())
            dist = np.stack(dist)                                                 # [loops,npairs,64]
            sdir.mkdir(parents=True, exist_ok=True)
            np.save(sdir / "cross_distogram_loops.npy", dist.astype(np.float16))

            iptms = [float(x.iptm if x.iptm is not None else (x.ptm or 0.0)) for x in samples]
            json.dump({"seed": s, "iptm_per_sample": iptms, "best_iptm": max(iptms),
                       "n_loops": len(zs), "npairs": int(dist.shape[1])},
                      open(sdir / "confidence.json", "w"))
            warn = "" if len(zs) == expected else f"  WARN got {len(zs)} loops"
            print(f"  best ipTM={max(iptms):.3f}  dist{tuple(dist.shape)}{warn}", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
