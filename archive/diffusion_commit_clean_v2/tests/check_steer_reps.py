#!/usr/bin/env python
"""End-to-end checks for steer_reps.py against a saved DUMP_FULL run (needs a GPU and LAYERNORM_TYPE=openfold).

    LAYERNORM_TYPE=openfold CUDA_VISIBLE_DEVICES=0 check_steer_reps.py --run <steer_..._full dir>

Each check prints PASS/FAIL with the numbers. Nothing here writes outside --out.
"""
import argparse, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import steer_reps as S

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--pdb", default="8w5q")
ap.add_argument("--seed", default="300019"); A = ap.parse_args()
R = A.run; dev = "cuda"; fails = []
def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"   {info}" if info else "")); ok or fails.append(name)

stem = glob.glob(f"{R}/full/*_reps.npz")[0][:-9]
order = json.load(open(f"{R}/sample_order.json"))["cif_sample_to_dump_index"]
cifs = sorted(glob.glob(f"{R}/out/{A.pdb}/{A.seed}-*/predictions/{A.pdb}_sample_?.cif"))
model, configs = S.build_model("/home/ebrahim/checkpoint", "protenix-v2", dev)
feat, z, s, si, coords = S.load_dump(stem, dev)
n_cycle = int(np.load(stem + "_reps.npz")["n_cycle"])

# 1. target resolution: CIF sample_0 must be dump index order['0']
idx, tgt = S.resolve_target(coords, None, cifs[0])
check("target cif sample_0 -> dump index from sample_order.json", idx == order["0"], f"got {idx}, expected {order['0']}")
dup = torch.cat([coords[0:1], coords[0:1], coords[1:]], 0)      # dump pose 0 twice; cif sample_4 == dump pose 0
try:
    S.resolve_target(dup, None, cifs[4])
    check("ambiguous target is refused", False, "should have raised")
except AssertionError:
    check("ambiguous target is refused", True)

# 2. confidence path reproduces the ORIGINAL poses' saved summary and pLDDT logits
x = coords.clone()
summ, plddt = S.confidence(model, configs, feat, si, s, z, x, n_cycle)
rep = np.load(stem + "_reps.npz")
dp = np.abs(plddt.float().cpu().numpy() - rep["plddt"]).max()
check("pLDDT logits from dumped z/s/features == dumped pLDDT", dp < 0.5, f"max|diff| {dp:.4f} (scale of logits ~{np.abs(rep['plddt']).max():.1f})")
for k, cif in enumerate(cifs):
    with open(cif.replace(f"{A.pdb}_sample_{k}.cif", f"{A.pdb}_summary_confidence_sample_{k}.json")) as fh:
        j = json.load(fh)
    d = order[str(k)]
    check(f"ipTM for cif sample_{k} (dump {d}) matches saved summary", abs(float(summ[d]["iptm"]) - j["iptm"]) < 0.01,
          f"recomputed {float(summ[d]['iptm']):.4f} saved {j['iptm']:.4f}")

# 3. loss: new (aligned MSE + lDDT) vs the first version's raw MSE
loss_fn = S.make_loss(model, configs, feat, si, tgt, 1.0, dev)
loss_mse_only = S.make_loss(model, configs, feat, si, tgt, 0.0, dev)
from protenix.model.generator import sample_diffusion_training, TrainingNoiseSampler
ns = TrainingNoiseSampler(**configs.train_noise_sampler.to_dict()); sd = float(configs.sigma_data)
def old_loss():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, xd, sg = sample_diffusion_training(noise_sampler=ns, denoise_net=model.diffusion_module,
            label_dict={"coordinate": tgt, "coordinate_mask": torch.ones(tgt.shape[0], device=dev)}, input_feature_dict=feat,
            s_inputs=si, s_trunk=s, z_trunk=z, pair_z=None, p_lm=None, c_l=None, N_sample=16)
    xd = xd.float(); sg = sg.float(); w = (sg**2 + sd**2) / ((sg*sd)**2)
    return float((w * ((xd - tgt.unsqueeze(0))**2).sum(-1).mean(-1)).mean())
torch.manual_seed(999); lo = old_loss()
torch.manual_seed(999)
with torch.no_grad(): ln, parts = loss_fn(z, s, 16)
check("old raw-MSE loss is frame-dominated (much larger than aligned loss)", lo > 20 * parts["mse"], f"old {lo:.3f}  new aligned mse {parts['mse']:.5f} lddt {parts['lddt']:.4f}")
# decoy target: the loss should be finite and the same order of magnitude for any pose the model makes
for d in range(coords.shape[0]):
    lf = S.make_loss(model, configs, feat, si, coords[d], 1.0, dev); torch.manual_seed(999)
    with torch.no_grad(): l, p = lf(z, s, 16)
    print(f"     info: unedited z/s, target = dump[{d}]: mse {p['mse']:.5f} lddt {p['lddt']:.4f}")
# rotation invariance of the loss in the TARGET's frame
th = 0.7; Rz = torch.tensor([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], dtype=torch.float32, device=dev)
lf_rot = S.make_loss(model, configs, feat, si, tgt @ Rz.T + 5.0, 1.0, dev); torch.manual_seed(999)
with torch.no_grad(): lr_, pr = lf_rot(z, s, 16)
check("loss unchanged when the target is rotated+translated", abs(pr["mse"] - parts["mse"]) < 0.3 * parts["mse"] + 1e-6,
      f"mse {parts['mse']:.5f} -> {pr['mse']:.5f}; lddt {parts['lddt']:.4f} -> {pr['lddt']:.4f} (bf16 noise + non-exact equivariance)")

# 4. optimisation: gradients flow to z and s, loss drops on a FIXED evaluation noise draw
def eval_loss(zz, ss):
    torch.manual_seed(4242)
    with torch.no_grad(): return float(loss_fn(zz, ss, 16)[0])
zp = z.clone().requires_grad_(True); sp = s.clone().requires_grad_(True); opt = torch.optim.Adam([zp, sp], lr=0.01)
before = eval_loss(zp, sp); torch.manual_seed(0)
for i in range(25):
    l, _ = loss_fn(zp, sp, 4); opt.zero_grad(); l.backward()
    if i == 0: check("nonzero gradient reaches z and s", zp.grad.norm() > 0 and sp.grad.norm() > 0, f"|gz| {zp.grad.norm():.3g} |gs| {sp.grad.norm():.3g}")
    opt.step()
after = eval_loss(zp, sp)
check("loss on a fixed noise draw decreases over 25 steps", after < before, f"{before:.5f} -> {after:.5f}; rel edit z {((zp-z).norm()/z.norm()).item():.2e} s {((sp-s).norm()/s.norm()).item():.2e}")
check("optimised state is finite", bool(torch.isfinite(zp).all() and torch.isfinite(sp).all()))

# 5. decode from the unedited state is finite and the confidence path runs on it
xd = S.decode(model, configs, feat, si, z, s, 5, 1234, dev); sm, _ = S.confidence(model, configs, feat, si, s, z, xd, n_cycle)
check("decode of unedited state finite; ipTM finite", bool(np.isfinite([float(m['iptm']) for m in sm]).all()), f"ipTM {[round(float(m['iptm']),3) for m in sm]}")
print("\nFAILED: " + ", ".join(fails) if fails else "\nALL CHECKS PASSED"); sys.exit(1 if fails else 0)
