#!/usr/bin/env python
"""Draw more diffusion poses from a SAVED trunk state (z_trunk, s_trunk, s_inputs, input features).

SUMMARY
    The trunk is deterministic (8w5q seed 300019: float32 z identical across two runs), so everything
    that varies between poses of one seed happens in the diffusion head. This loads the dump written by
    Protenix DUMP_FULL, runs ONLY the diffusion sampler N times from it, and writes the poses.

    Precision matches the original fold: sampling under bf16 autocast (folder.py default --dtype bf16).
    pair_z is passed as None so the sampler rebuilds it from z_trunk, as in steer_reps.py.

    Poses come out in the same atom order as the fold's CIFs (verified: dumped coordinate == CIF
    coordinates to 0.000 A), so each is written as a CIF by putting the new coordinates into a template
    CIF of the same complex.

Usage:
  sample_from_trunk.py --dump <run>/full/8w5q_free_seed300019 --template-cif <run>/out/.../8w5q_sample_0.cif \
                       --n 10 --seed 7 --out <run>/resampled
"""
import argparse
import json
import os
import sys
import time

import gemmi
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steer_reps import build_model, load_dump  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="stem, without _reps.npz / _feat.npz")
    ap.add_argument("--template-cif", required=True, help="a CIF of this complex, for atom layout")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, required=True, help="torch seed for the noise draws; recorded")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    dev = "cuda"

    model, configs = build_model(A.ckpt_dir, A.model_name, dev)
    feat, z, s, s_inputs, coords0 = load_dump(A.dump, dev)
    print(f"tokens {z.shape[0]}  atoms {coords0.shape[1]}  dumped poses {coords0.shape[0]}")

    from protenix.model.generator import InferenceNoiseScheduler
    sched = InferenceNoiseScheduler(**configs.inference_noise_scheduler.to_dict())
    noise_schedule = sched(N_step=configs.sample_diffusion.N_step, device=dev)

    torch.manual_seed(A.seed)
    t0 = time.time()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x = model.sample_diffusion(
            denoise_net=model.diffusion_module, input_feature_dict=feat,
            s_inputs=s_inputs, s_trunk=s, z_trunk=z,      # no batch dim, as protenix.py passes them
            pair_z=None, p_lm=None, c_l=None, N_sample=A.n,
            noise_schedule=noise_schedule, inplace_safe=False)
    xa = x.squeeze(0).float().cpu().numpy()
    assert xa.shape == coords0.shape[1:] or xa.shape[0] == A.n, xa.shape
    assert np.isfinite(xa).all(), "non-finite coordinates"
    print(f"sampled {xa.shape} in {time.time()-t0:.0f}s")
    np.savez(f"{A.out}/coordinate.npz", coordinate=xa, seed=np.array(A.seed))

    st0 = gemmi.read_structure(A.template_cif)
    n_atoms = sum(1 for ch in st0[0] for r in ch for a in r)
    assert n_atoms == xa.shape[1], (n_atoms, xa.shape)
    for k in range(A.n):
        st = st0.clone()
        i = 0
        for ch in st[0]:
            for r in ch:
                for a in r:
                    a.pos = gemmi.Position(*[float(v) for v in xa[k, i]])
                    i += 1
        st.make_mmcif_document().write_file(f"{A.out}/resampled_{k:02d}.cif")
    json.dump(dict(dump=A.dump, template_cif=A.template_cif, n=A.n, seed=A.seed,
                   dtype="bf16 autocast", n_step=int(configs.sample_diffusion.N_step),
                   note="poses in dump atom order; resampled_kk.cif = coordinate[kk]"),
              open(f"{A.out}/meta.json", "w"), indent=1)
    print(f"wrote {A.n} poses to {A.out}")


if __name__ == "__main__":
    main()
