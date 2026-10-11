#!/usr/bin/env python
"""Optimise a saved trunk state (z_trunk + s_trunk) so the diffusion module reproduces ONE target pose.

THE QUESTION. For a seed whose diffusion samples land in several docking arrangements from a single
frozen trunk state, can z/s be edited so that the sampler commits to the target arrangement every
time, and how large an edit does that take? The target is a pose the seed ALREADY produces (a
superposition being resolved), not a pose the model never held.

PARAMETERISATION. The default optimises z_trunk and s_trunk DIRECTLY (unconstrained, every entry free).
`--mode pairformer-lora` instead freezes the pretrained model, adds low-rank adapters to every Linear
inside PairformerStack, and optimises only those adapter weights. In that mode the saved z/s each pass
through one full PairformerStack (all configured blocks, with no extra MSA/template/recycle projections)
before entering diffusion. The adapted z/s are saved at each checkpoint together with LoRA weights.

THE LOSS is Protenix's own training loss for the diffusion module, restricted to the terms that make
sense here, one denoising step, no trunk in the graph (generator.py sample_diffusion_training):

    1. pick a noise level sigma ~ the TRAINING distribution (sigma_data * exp(p_mean + p_std * n)),
       corrupt a randomly rotated+translated copy of the target with it, and denoise ONCE
       (--n-sample-loss independent draws per step);
    2. MSE term (loss.py MSELoss): the TARGET is rigidly aligned onto the denoised structure (weighted
       Kabsch, gradient stopped), then the per-atom squared error is taken, then multiplied by
       (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2 -- the EDM weight, large at small sigma;
    3. smooth-lDDT term (loss.py SmoothLDDTLoss): frame-independent, rewards local distances within
       15 A of each atom, so interface geometry counts. Weighted by --lddt-weight (default 1.0, the
       value in configs_base.py). Zero switches it off.
    total = MSE + lddt_weight * smooth-lDDT.  (Protenix multiplies both by alpha_diffusion = 4; Adam is
    scale-invariant so that constant is omitted. The bond term has weight 0 in configs_base.py.)

    WHY THE ALIGNMENT MATTERS. sample_diffusion_training returns x_denoised in the frame of the RANDOMLY
    ROTATED copy it noised. Comparing that with the un-rotated target (as the first version of this
    script did) measures the random rotation, not the pose. MSELoss removes the frame by aligning.

pair_z IS DELIBERATELY None so the diffusion module recomputes it from z_trunk INSIDE the graph; a
cached pair_z would detach the leaf and silently optimise nothing (checked by tests/check_steer_reps.py).

SHAPES. z [N,N,c_z], s [N,c_s], s_inputs [N,c_si] with NO batch dimension, exactly as
protenix.py passes them; the input feature dict has none either, so adding one crashes in
prepare_cache (got 4 and 3 dims).

PRECISION. Optimisation and decoding both run under bf16 autocast, as folder.py --dtype bf16 does (the
original folds were bf16). Parameters and checkpoints are float32: an edited z is not bf16-valued, and
the first version saved it as float16, which quantises away small edits before decoding.

REPORTED PER CHECKPOINT: relative edit size ||dz||/||z||, and optionally the distance between two REAL
trunk states of the same complex (--ref-trunk, a second seed) as a yardstick. An edit far below that
is persuasion; far above it is overwriting.

Usage:
  LAYERNORM_TYPE=openfold steer_reps.py --dump <run>/full/8w5q_free_seed300019 \
        --target-cif <run>/out/8w5q/300019-*/predictions/8w5q_sample_0.cif \
        --ckpts 0,10,25,50,100 --decode --template-cif <same cif> --out <dir>

  LAYERNORM_TYPE=openfold steer_reps.py --mode pairformer-lora --dump <stem> \
        --target-cif <pose.cif> --lr-z <lr> --lr-s <lr> --ckpts 0,10,25 \
        --decode --template-cif <template.cif> --out <dir>
"""
import argparse
import functools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/ebrahim/Protenix")
print = functools.partial(print, flush=True)      # the log is read live; never leave it in a buffer


def build_model(ckpt_dir, model_name, device):
    """Instantiate Protenix exactly as runner/inference.py does, then load the checkpoint.

    The config assembly is replicated rather than imported because inference.py performs it inside
    run(), which also parses argv and launches a full inference job.
    """
    # Without openfold layernorm the model emits NaN coordinates silently (folder.py docstring).
    if os.environ.get("LAYERNORM_TYPE") != "openfold":
        raise SystemExit("set LAYERNORM_TYPE=openfold (without it coordinates are silently NaN)")
    from collections.abc import Mapping
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config import parse_configs
    from protenix.model.protenix import Protenix
    try:
        from runner.inference import update_gpu_compatible_configs, model_configs
    except ImportError:
        from runner.inference import model_configs
        update_gpu_compatible_configs = lambda c: c

    base = {**configs_base, **{"data": data_configs}, **inference_configs}

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d
    deep_update(base, model_configs[model_name])
    configs = parse_configs(configs=base, arg_str="", fill_required_with_null=True)
    configs.load_checkpoint_dir = ckpt_dir
    configs.model_name = model_name
    configs = update_gpu_compatible_configs(configs)

    model = Protenix(configs).to(device)
    ck = torch.load(os.path.join(ckpt_dir, f"{model_name}.pt"), map_location=device,
                    weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    print(f"checkpoint: {len(res.missing_keys)} missing keys, {len(res.unexpected_keys)} unexpected")
    if res.missing_keys:
        raise SystemExit(f"checkpoint is missing {len(res.missing_keys)} weights, e.g. "
                         f"{res.missing_keys[:5]} -- the model would run partly random")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)          # only z/s are optimised
    return model, configs


class LoRAWeight(nn.Module):
    """Low-rank weight residual; parametrization also covers fused direct-weight call sites."""
    def __init__(self, base_weight, rank=8, alpha=16.0):
        super().__init__()
        self.scale = float(alpha) / int(rank)
        self.lora_A = nn.Parameter(torch.empty(int(rank), base_weight.shape[1],
                                               device=base_weight.device, dtype=base_weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base_weight.shape[0], int(rank),
                                               device=base_weight.device, dtype=base_weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

    def forward(self, base_weight):
        return base_weight + (self.lora_B @ self.lora_A) * self.scale


def add_pairformer_lora(model, rank=8, alpha=16.0):
    """Freeze the pretrained model and add LoRA to every Linear in PairformerStack."""
    for p in model.parameters():
        p.requires_grad_(False)
    stack = model.pairformer_stack
    targets = [(name, module) for name, module in stack.named_modules()
               if name and isinstance(module, nn.Linear)]
    params = {"z": [], "s": []}
    for name, module in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = stack.get_submodule(parent_name) if parent_name else stack
        module.weight.requires_grad_(False)
        if module.bias is not None:
            module.bias.requires_grad_(False)
        parametrization = LoRAWeight(module.weight, rank=rank, alpha=alpha)
        torch.nn.utils.parametrize.register_parametrization(module, "weight", parametrization)
        module.parametrizations.weight.original.requires_grad_(False)
        # Pair/single update paths retain the existing --lr-z / --lr-s distinction.
        group = "s" if ("attention_pair_bias" in name or "single_transition" in name) else "z"
        params[group].extend((parametrization.lora_A, parametrization.lora_B))
    if not targets or not params["z"] or not params["s"]:
        raise RuntimeError("No Pairformer linear layers were adapted")
    print(f"installed LoRA rank={rank}, alpha={alpha} on "
          f"{len(targets)} Pairformer linear layers "
          f"({sum(p.numel() for group in params.values() for p in group):,} trainable parameters)")
    return params


def pairformer_once(model, configs, s, z):
    """One full PairformerStack pass (one stack of blocks), with no extra recycle modules."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model.pairformer_stack(
            s, z, pair_mask=None,
            triangle_multiplicative=configs.triangle_multiplicative,
            triangle_attention=configs.triangle_attention,
            inplace_safe=False, chunk_size=None)


def load_dump(stem, device):
    """Read <stem>_reps.npz + <stem>_feat.npz written by Protenix DUMP_FULL.
    Returns feat, z [N,N,c_z], s [N,c_s], s_inputs [N,c_si], coords [n_pose,N_atom,3] (DUMP order,
    which is NOT the order of the *_sample_k.cif files -- those are sorted by ranking score)."""
    from protenix.model.protenix import update_input_feature_dict
    R = np.load(f"{stem}_reps.npz")
    F = np.load(f"{stem}_feat.npz")
    feat = {}
    for k in F.files:
        v = F[k]
        if v.dtype == np.bool_:
            feat[k] = torch.from_numpy(v).to(device)
        elif np.issubdtype(v.dtype, np.integer):
            feat[k] = torch.from_numpy(v).long().to(device)
        else:
            feat[k] = torch.from_numpy(v).float().to(device)
    # d_lm / v_lm / pad_info are rebuilt rather than trusted from the dump: pad_info was never saved,
    # and all three are pure functions of ref_pos + ref_space_uid (protenix.py:55).
    feat = update_input_feature_dict(feat)
    for k in ("d_lm", "v_lm", "pad_info", "is_dna", "is_rna", "is_ligand", "asym_id"):
        assert k in feat, f"feature {k} missing after load"
    z = torch.from_numpy(R["z_trunk"]).float().to(device)
    s = torch.from_numpy(R["s_trunk"]).float().to(device)
    si = torch.from_numpy(R["s_inputs"]).float().to(device)
    coords = torch.from_numpy(R["coordinate"]).float().to(device)
    assert z.dim() == 3 and s.dim() == 2 and si.dim() == 2 and coords.dim() == 3, \
        (z.shape, s.shape, si.shape, coords.shape)
    assert z.shape[0] == z.shape[1] == s.shape[0] == si.shape[0]
    for name, t in (("z", z), ("s", s), ("s_inputs", si), ("coordinate", coords)):
        assert torch.isfinite(t).all(), f"{name} has non-finite values"
    return feat, z, s, si, coords


def cif_coords(path):
    """Atom coordinates of a CIF in file order (== the model's atom order, verified 0.000 A)."""
    import gemmi
    return np.array([a.pos.tolist() for ch in gemmi.read_structure(path)[0] for r in ch for a in r])


def resolve_target(coords, target_sample, target_cif):
    """Pick the target pose. --target-cif finds the dump index by EXACT coordinate match (and refuses
    unless exactly one dumped pose matches), so the CIF-vs-dump ordering can not be got wrong."""
    if target_cif:
        c = torch.from_numpy(cif_coords(target_cif)).float().to(coords.device)
        assert c.shape == coords.shape[1:], (c.shape, coords.shape)
        d = (coords - c.unsqueeze(0)).abs().amax(dim=(1, 2)).cpu().numpy()
        hits = np.where(d < 1e-2)[0]          # CIF text rounds to 3 decimals
        assert len(hits) == 1, f"target cif matches {len(hits)} dumped poses (max diffs {d.round(4)})"
        idx = int(hits[0])
        print(f"target cif -> dump index {idx} (max coordinate diff {d[idx]:.4f} A)")
    else:
        assert target_sample is not None, "give --target-cif or --target-sample"
        idx = target_sample
        assert 0 <= idx < coords.shape[0], idx
    return idx, coords[idx]


def make_loss(model, configs, feat, s_inputs, tgt, lddt_weight, dev):
    """Return loss_fn(z, s, n_draws) -> (total, parts). See the module docstring."""
    from protenix.model.generator import sample_diffusion_training, TrainingNoiseSampler
    from protenix.model.loss import MSELoss, SmoothLDDTLoss, compute_lddt_mask
    from protenix.utils.torch_utils import cdist
    # configs.train_noise_sampler, NOT configs.loss.*: p_mean -1.2, p_std 1.5, sigma_data 16.0
    noise_sampler = TrainingNoiseSampler(**configs.train_noise_sampler.to_dict())
    mse = MSELoss(**configs.loss.diffusion.mse)
    coord_mask = torch.ones(tgt.shape[0], device=dev)
    label = {"coordinate": tgt, "coordinate_mask": coord_mask}
    sd = float(configs.sigma_data)
    lddt = lmask = None
    if lddt_weight > 0:
        lddt = SmoothLDDTLoss(**configs.loss.diffusion.smooth_lddt)
        dmask = coord_mask[:, None] * coord_mask[None, :]
        lmask = compute_lddt_mask(true_distance=cdist(tgt, tgt) * dmask, distance_mask=dmask,
                                  is_nucleotide=(feat["is_dna"] + feat["is_rna"]).bool(),
                                  is_nucleotide_threshold=30.0, is_not_nucleotide_threshold=15.0)

    def loss_fn(z, s, n_draws):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _x_gt_augmented_unused, x_den, sigma = sample_diffusion_training(
                noise_sampler=noise_sampler, denoise_net=model.diffusion_module,
                label_dict=label, input_feature_dict=feat,
                s_inputs=s_inputs, s_trunk=s, z_trunk=z,
                pair_z=None, p_lm=None, c_l=None, N_sample=n_draws)
        # x_den is in the frame of the augmented copy; MSELoss aligns the target onto it.
        x_den = x_den.float()
        sigma = sigma.float()
        scale = (sigma ** 2 + sd ** 2) / ((sd * sigma) ** 2)
        l_mse = mse(pred_coordinate=x_den, true_coordinate=tgt, coordinate_mask=coord_mask,
                    is_dna=feat["is_dna"], is_rna=feat["is_rna"], is_ligand=feat["is_ligand"],
                    per_sample_scale=scale)
        total = l_mse
        parts = dict(mse=float(l_mse.detach()), sigma_mean=float(sigma.mean()))
        if lddt is not None:
            l_l = lddt.dense_forward(pred_coordinate=x_den, true_coordinate=tgt, lddt_mask=lmask,
                                     diffusion_chunk_size=getattr(configs.loss, "diffusion_lddt_chunk_size", None))
            total = total + lddt_weight * l_l
            parts["lddt"] = float(l_l.detach())
        return total, parts
    return loss_fn


def decode(model, configs, feat, s_inputs, z, s, n, seed, dev):
    """The normal 200-step sampler from a given trunk state; returns coordinates [n,N_atom,3].
    pair_z=None so it is rebuilt from the (edited) z."""
    from protenix.model.generator import InferenceNoiseScheduler
    sched = InferenceNoiseScheduler(**configs.inference_noise_scheduler.to_dict())
    noise_schedule = sched(N_step=configs.sample_diffusion.N_step, device=dev)
    torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x = model.sample_diffusion(
            denoise_net=model.diffusion_module, input_feature_dict=feat,
            s_inputs=s_inputs, s_trunk=s, z_trunk=z, pair_z=None, p_lm=None, c_l=None,
            N_sample=n, noise_schedule=noise_schedule, inplace_safe=False)
    x = x.float()
    assert x.dim() == 3 and torch.isfinite(x).all(), "sampler returned non-finite coordinates"
    return x


def confidence(model, configs, feat, s_inputs, s, z, x, n_cycle):
    """ipTM / ranking score / pLDDT logits for poses x, exactly as protenix.py computes them.
    Returns (summary, plddt_logits). Validated against the saved summary jsons in tests/."""
    from protenix.model import sample_confidence
    from protenix.utils.torch_utils import autocasting_disable_decorator
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        plddt, pae, pde, _resolved = model.run_confidence_head(
            input_feature_dict=feat, s_inputs=s_inputs, s_trunk=s, z_trunk=z, pair_mask=None,
            x_pred_coords=x, triangle_multiplicative=configs.triangle_multiplicative,
            triangle_attention=configs.triangle_attention, inplace_safe=False, chunk_size=None)
        cp = autocasting_disable_decorator(True)(sample_confidence.compute_contact_prob)(
            distogram_logits=model.distogram_head(z),
            **sample_confidence.get_bin_params(configs.loss.distogram))
        summary, _full = autocasting_disable_decorator(True)(
            sample_confidence.compute_full_data_and_summary)(
            configs=configs, pae_logits=pae, plddt_logits=plddt, pde_logits=pde, contact_probs=cp,
            token_asym_id=feat["asym_id"], token_has_frame=feat["has_frame"], atom_coordinate=x,
            atom_to_token_idx=feat["atom_to_token_idx"], atom_is_polymer=1 - feat["is_ligand"],
            N_recycle=n_cycle, interested_atom_mask=None, return_full_data=True,
            mol_id=None, elements_one_hot=None)
    return summary, plddt


def write_cifs(template_cif, xa, out_dir, prefix):
    import gemmi
    st0 = gemmi.read_structure(template_cif)
    assert sum(1 for ch in st0[0] for r in ch for a in r) == xa.shape[1]
    os.makedirs(out_dir, exist_ok=True)
    for k in range(xa.shape[0]):
        st = st0.clone()
        i = 0
        for ch in st[0]:
            for r in ch:
                for a in r:
                    a.pos = gemmi.Position(*[float(v) for v in xa[k, i]])
                    i += 1
        st.make_mmcif_document().write_file(f"{out_dir}/{prefix}_{k:02d}.cif")


def kabsch_rmsd(a, b):
    """Whole-complex RMSD after optimal superposition. A rough diagnostic only: DockQ against the
    crystal (scorer.py) is the real measure."""
    a = a - a.mean(0); b = b - b.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt(((a @ R - b) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="stem, without _reps.npz / _feat.npz")
    ap.add_argument("--target-cif", default="", help="CIF of the target pose (preferred)")
    ap.add_argument("--target-sample", type=int, default=None,
                    help="index into the DUMPED coordinate array (NOT the CIF sample number)")
    ap.add_argument("--ckpts", default="0,100,200")
    ap.add_argument("--lr-z", type=float, required=True, help="Adam lr for z; a step is lr, so compare with z's entry size (~20 for 8w5q)")
    ap.add_argument("--lr-s", type=float, required=True, help="Adam lr for s (entry size ~220 for 8w5q)")
    ap.add_argument("--betas", default="0.9,0.95", help="Protenix training betas")
    ap.add_argument("--eps", type=float, default=1e-12, help="Adam eps; Protenix leaves 1e-8, but z's per-entry gradient is ~2e-9")
    ap.add_argument("--warmup", type=int, default=10, help="linear lr warmup steps (Protenix: 10)")
    ap.add_argument("--grad-clip", type=float, default=10.0, help="global grad-norm clip (Protenix: 10)")
    ap.add_argument("--loss-scale", type=float, default=4.0, help="alpha_diffusion; Protenix multiplies the diffusion loss by 4")
    ap.add_argument("--mode", choices=("direct", "pairformer-lora"), default="direct",
                    help="edit z/s directly (default), or train LoRA adapters in one full PairformerStack pass")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--micro-batch", type=int, default=8, help="noise draws per forward/backward; grads are accumulated up to --n-sample-loss")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--n-eval", type=int, default=32, help="size of the FIXED evaluation set of noise draws")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (default: largest --ckpts)")
    ap.add_argument("--lddt-weight", type=float, default=1.0)
    ap.add_argument("--n-sample-loss", type=int, default=48, help="noise draws per optimisation step (Protenix diffusion_batch_size: 48)")
    ap.add_argument("--ref-trunk", default="", help="npz with z_trunk/s_trunk of another real seed, for the edit-size yardstick")
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--seed", type=int, default=0, help="seed for the optimisation noise draws")
    ap.add_argument("--out", required=True)
    ap.add_argument("--decode", action="store_true", help="decode poses at each checkpoint (off: get the loss down first)")
    ap.add_argument("--n-sample-decode", type=int, default=20)
    ap.add_argument("--decode-seed", type=int, default=1234,
                    help="noise seed for decoding; same at every checkpoint, different from --seed")
    ap.add_argument("--template-cif", default="", help="with --decode: write decoded poses as CIFs")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    dev = "cuda"
    assert A.decode_seed != A.seed, "decode noise must differ from the optimisation noise"

    model, configs = build_model(A.ckpt_dir, A.model_name, dev)
    feat, z0, s0, s_inputs, coords = load_dump(A.dump, dev)
    idx, tgt = resolve_target(coords, A.target_sample, A.target_cif)
    n_cycle = int(np.load(f"{A.dump}_reps.npz")["n_cycle"])
    print(f"tokens {z0.shape[0]}  atoms {tgt.shape[0]}  c_z {z0.shape[-1]}  c_s {s0.shape[-1]}  target=dump[{idx}]")

    zn = z0.norm().item(); sn = s0.norm().item()
    yard = None
    if A.ref_trunk:
        R = np.load(A.ref_trunk)
        zr = torch.from_numpy(R["z_trunk"]).float().to(dev); sr = torch.from_numpy(R["s_trunk"]).float().to(dev)
        assert zr.shape == z0.shape and sr.shape == s0.shape, "ref trunk is a different complex"
        yard = dict(rel_z=(z0 - zr).norm().item() / zn, rel_s=(s0 - sr).norm().item() / sn)
        print(f"YARDSTICK two real trunk states: ||dz||/||z|| = {yard['rel_z']:.5f}   ||ds||/||s|| = {yard['rel_s']:.5f}")
        del zr, sr

    loss_fn = make_loss(model, configs, feat, s_inputs, tgt, A.lddt_weight, dev)
    assert A.n_sample_loss % A.micro_batch == 0 and A.n_eval % A.micro_batch == 0, "micro-batch must divide both draw counts"
    adapter_params = None
    if A.mode == "pairformer-lora":
        if A.lora_rank < 1 or A.lora_alpha <= 0:
            raise SystemExit("LoRA requires rank >= 1 and alpha > 0")
        adapter_params = add_pairformer_lora(model, A.lora_rank, A.lora_alpha)
        train_params = adapter_params["z"] + adapter_params["s"]
    else:
        zp = z0.clone().requires_grad_(True)
        sp = s0.clone().requires_grad_(True)
        train_params = [zp, sp]

    def trunk_state():
        if A.mode == "direct":
            return sp, zp
        # The saved representations enter one full PairformerStack pass; the pretrained
        # Pairformer weights stay frozen, and gradients update only its LoRA adapters.
        return pairformer_once(model, configs, s0, z0)

    b1, b2 = (float(x) for x in A.betas.split(","))
    if A.mode == "direct":
        opt = torch.optim.Adam([{"params": [zp], "lr": A.lr_z}, {"params": [sp], "lr": A.lr_s}],
                               betas=(b1, b2), eps=A.eps, weight_decay=0.0)
    else:
        opt = torch.optim.Adam([{"params": adapter_params["z"], "lr": A.lr_z},
                                {"params": adapter_params["s"], "lr": A.lr_s}],
                               betas=(b1, b2), eps=A.eps, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / max(A.warmup, 1)))
    n_trainable = sum(p.numel() for p in train_params)
    print(f"optimising {n_trainable:,} parameters (mode={A.mode}): lr_z {A.lr_z} lr_s {A.lr_s} betas {A.betas} "
          f"eps {A.eps} warmup {A.warmup} clip {A.grad_clip} loss x{A.loss_scale}; "
          f"{A.n_sample_loss} draws/step in micro-batches of {A.micro_batch}")

    def eval_fixed():
        """Loss on a FIXED set of A.n_eval noise draws (same sigmas, noise, rotations every call), no grad.
        The training loss changes step to step mostly because different noise levels are drawn; this does not."""
        tot = mse = ld = 0.0
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(A.seed + 10_000)
            s_eval, z_eval = trunk_state()
            for _ in range(A.n_eval // A.micro_batch):
                l, p = loss_fn(z_eval, s_eval, A.micro_batch)
                tot += float(l); mse += p["mse"]; ld += p.get("lddt", 0.0)
        k = A.n_eval // A.micro_batch
        return dict(total=tot / k, mse=mse / k, lddt=ld / k)

    ckpts = sorted({int(x) for x in A.ckpts.split(",")})
    max_steps = A.max_steps if A.max_steps is not None else max(ckpts)
    hist, evals = [], []
    torch.manual_seed(A.seed)

    def save(step, ev):
        with torch.no_grad(): s_out, z_out = trunk_state()
        rel_z = (z_out - z0).norm().item() / zn
        rel_s = (s_out - s0).norm().item() / sn
        np.savez(f"{A.out}/ckpt{step:04d}.npz", z_trunk=z_out.detach().cpu().numpy(),
                 s_trunk=s_out.detach().cpu().numpy(), step=np.array(step),
                 rel_z=np.array(rel_z), rel_s=np.array(rel_s))
        if A.mode == "pairformer-lora":
            adapter_state = {k: v.detach().cpu() for k, v in model.pairformer_stack.state_dict().items()
                             if "lora_A" in k or "lora_B" in k}
            torch.save(adapter_state, f"{A.out}/lora_adapters{step:04d}.pt")
        hist.append(dict(step=step, rel_z=rel_z, rel_s=rel_s, fixed_eval=ev))
        print(f"   [ckpt {step:>4}] ||dz||/||z|| {rel_z:.5f}   ||ds||/||s|| {rel_s:.5f}   fixed-eval {ev}")

    t0 = time.time()
    for step in range(max_steps + 1):
        ev = None
        if step % A.eval_every == 0 or step == max_steps or step in ckpts:
            ev = eval_fixed()
            with torch.no_grad(): s_eval, z_eval = trunk_state()
            rel_z = (z_eval - z0).norm().item() / zn; rel_s = (s_eval - s0).norm().item() / sn
            evals.append(dict(step=step, rel_z=rel_z, rel_s=rel_s, **ev))
            print(f"EVAL step {step:>4}  fixed-{A.n_eval}-draw loss total {ev['total']:.5f}  mse {ev['mse']:.5f}  lddt {ev['lddt']:.5f}   "
                  f"||dz||/||z|| {rel_z:.5f} ||ds||/||s|| {rel_s:.5f}")
        if step in ckpts:
            save(step, ev)
        if step == max_steps:
            break
        opt.zero_grad(set_to_none=True)
        n_mb = A.n_sample_loss // A.micro_batch
        tr_total = tr_mse = tr_lddt = 0.0
        s_step, z_step = trunk_state()
        for _ in range(n_mb):
            l, p = loss_fn(z_step, s_step, A.micro_batch)
            (A.loss_scale * l / n_mb).backward(retain_graph=(A.mode == "pairformer-lora" and _ < n_mb - 1))
            tr_total += float(l) / n_mb; tr_mse += p["mse"] / n_mb; tr_lddt += p.get("lddt", 0.0) / n_mb
        if A.mode == "direct":
            gz, gs = zp.grad.norm().item(), sp.grad.norm().item()
            grad_ok = np.isfinite(gz) and np.isfinite(gs)
            grad_norm = float(torch.linalg.vector_norm(torch.stack([zp.grad.norm(), sp.grad.norm()])))
        else:
            norms = [p.grad.norm() for p in train_params if p.grad is not None]
            grad_norm = float(torch.linalg.vector_norm(torch.stack(norms))) if norms else 0.0
            grad_ok = np.isfinite(grad_norm)
            gz, gs = grad_norm, 0.0
        if not (grad_ok and np.isfinite(tr_total)):
            raise SystemExit(f"step {step}: non-finite loss/gradient (loss {tr_total}, grad norm {grad_norm})")
        if step == 0:
            assert grad_norm > 0, "zero gradient: the loss is detached from the trainable parameters"
            print(f"   peak GPU memory after first step: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
        torch.nn.utils.clip_grad_norm_(train_params, A.grad_clip)
        opt.step(); sched.step()
        grad_label = f"|grad z| {gz:.3g} |grad s| {gs:.3g}" if A.mode == "direct" else f"adapter grad norm {grad_norm:.3g}"
        print(f"   step {step:>4}  train loss {tr_total:.4f} (mse {tr_mse:.4f} lddt {tr_lddt:.4f})  {grad_label}  "
              f"lr {opt.param_groups[0]['lr']:.3g}  ({(time.time()-t0)/(step+1):.1f}s/step)")

    if A.decode:
        dec = []
        t_np = tgt.cpu().numpy()
        for step in ckpts:
            C = np.load(f"{A.out}/ckpt{step:04d}.npz")
            zc = torch.from_numpy(C["z_trunk"]).float().to(dev)
            sc = torch.from_numpy(C["s_trunk"]).float().to(dev)
            x = decode(model, configs, feat, s_inputs, zc, sc, A.n_sample_decode, A.decode_seed, dev)
            summ, _ = confidence(model, configs, feat, s_inputs, sc, zc, x, n_cycle)
            xa = x.cpu().numpy()
            d = [kabsch_rmsd(xa[k], t_np) for k in range(xa.shape[0])]
            np.savez(f"{A.out}/decode{step:04d}.npz", coordinate=xa, rmsd_to_target=np.array(d))
            if A.template_cif:
                write_cifs(A.template_cif, xa, f"{A.out}/decode{step:04d}", "pose")
            rec = dict(step=int(step), rmsd_to_target=d, rmsd_mean=float(np.mean(d)),
                       rmsd_spread=float(max(d) - min(d)),
                       iptm=[float(m["iptm"]) for m in summ], ranking_score=[float(m["ranking_score"]) for m in summ])
            dec.append(rec)
            print(f"   [decode {step:>4}] RMSD to target mean {np.mean(d):6.2f} A  spread {max(d)-min(d):5.2f}  "
                  f"ipTM mean {np.mean(rec['iptm']):.3f}")
        json.dump(dec, open(f"{A.out}/decode.json", "w"), indent=1)

    json.dump(dict(history=hist, fixed_eval_curve=evals, yardstick=yard, target_dump_index=idx, args=vars(A)),
              open(f"{A.out}/history.json", "w"), indent=1)
    print(f"\nwrote {len(ckpts)} checkpoints to {A.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
