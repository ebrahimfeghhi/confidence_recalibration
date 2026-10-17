"""GATE: an interface-plus-margin crop is INDEX-CONSISTENT, and the head and loss accept it.

WHAT THIS DOES AND DOES NOT ASSERT. It does NOT assert the loss is unchanged -- it cannot be, since
a crop has fewer token pairs, and the whole point is to compute on fewer. What can silently break is
the RE-INDEXING: atom_to_token_idx must point into the new token numbering and frame_atom_index into
the new atom numbering, or a pose is scored against another residue's frame. That is the failure
check_alignment exists to catch, on a path where check_alignment no longer runs, so it is checked
here instead -- element by element against the full arrays, which is a stronger test than a scalar
loss comparison would be.

Then it runs the real head and the real loss on the crop and requires finite values, and reports the
retained fraction with the projected speedup at the measured tokens^2.21.

  CUDA_VISIBLE_DEVICES=5 LAYERNORM_TYPE=openfold python -I phases/phase_k_crop_consistency.py
"""
import argparse, csv, os, random, sys, time
import numpy as np
import torch
# Protenix's pairformer calls torch.utils.checkpoint.checkpoint when blocks_per_ckpt is set, and
# torch.utils does not expose the submodule unless something imports it. The trainer gets it for
# free from another import; a gate that only ran forward under no_grad never reached the call.
import torch.utils.checkpoint  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")
import phase_j_build_cache as BC
import phase_j_rescore_cache as RC
from protenix.model.loss import ProtenixLoss

TERMS = ("plddt_loss", "pde_loss", "pae_loss", "resolved_loss")
# The same fold roots the trainer reads, in the same order.
ROOTS = ("training/fold", "training/fold_regen", "training/fold_deferred")
SEEDS = tuple(f"free_5000{i:02d}" for i in range(15))


def run_one_pose(model, configs, loss_fn, fh, s_in, s_tr, zt, poses, fl, co, cm, pi, dev,
                 backward=False):
    """One pose through the real head and the real loss, exactly as phase_k_train runs it.

    Returns (cumulative loss, {term: value}). With backward=True it also runs .backward(), which is
    the half of the computation a no_grad gate never touches."""
    x = poses[pi:pi + 1]
    ctx = torch.enable_grad() if backward else torch.no_grad()
    with ctx:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            plddt, pae, pde, resolved = model.run_confidence_head(
                input_feature_dict=fh, s_inputs=s_in, s_trunk=s_tr, z_trunk=zt,
                pair_mask=None, x_pred_coords=x,
                triangle_multiplicative=configs.triangle_multiplicative,
                triangle_attention=configs.triangle_attention,
                inplace_safe=False, chunk_size=None)
        pred = {"plddt": plddt.float(), "pae": pae.float(), "pde": pde.float(),
                "resolved": resolved.float(), "coordinate_mini": x}
        ld = {"coordinate": torch.as_tensor(co[pi], dtype=torch.float32, device=dev),
              "coordinate_mask": torch.as_tensor(cm[pi].astype(np.float32), device=dev)}
        cum, met = loss_fn.calculate_losses(fl, pred, ld, mode="train")
        if backward:
            cum.backward()
    return float(cum), {t: float(met[t]) for t in TERMS}


def time_fwd_bwd(model, configs, loss_fn, build, dev, reps=3):
    """Median wall-clock seconds for one pose's forward+backward. `build` returns the tensor set.

    MEASURED, NOT PROJECTED. The 2.21 exponent came from this project's own trajectory and predicts
    the head's cost well in aggregate, but a crop also changes the memory traffic and the kernel
    shapes, so the only honest speedup number is a timed one."""
    fh, s_in, s_tr, zt, poses, _nc, fl, (co, cm), _nb = build()
    ts = []
    for r in range(reps + 1):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_one_pose(model, configs, loss_fn, fh, s_in, s_tr, zt, poses, fl, co, cm, 0, dev,
                     backward=True)
        torch.cuda.synchronize()
        if r:                      # first pass is warm-up: autotuning and allocator growth
            ts.append(time.perf_counter() - t0)
    model.zero_grad(set_to_none=True)
    del fh, s_in, s_tr, zt, poses, fl, co, cm
    torch.cuda.empty_cache()
    return float(np.median(ts))


def check_indices(z, lab, keep_tok, keep_atom, sliced, pdb):
    """Every sliced array must equal the full array indexed by the kept mask. Returns failures."""
    feat_head, s_inputs, s_trunk, z_trunk, poses, _nc, feat_loss, (coords, cmask), n_broken = sliced
    kt, ka = np.nonzero(keep_tok)[0], np.nonzero(keep_atom)[0]
    tok_new = np.full(keep_tok.shape[0], -1, np.int64); tok_new[kt] = np.arange(kt.size)
    atom_new = np.full(keep_atom.shape[0], -1, np.int64); atom_new[ka] = np.arange(ka.size)
    bad = []

    def eq(name, got, want):
        g = got.detach().cpu().numpy() if torch.is_tensor(got) else np.asarray(got)
        w = np.asarray(want)
        if g.shape != w.shape:
            bad.append(f"{name}: shape {g.shape} vs {w.shape}"); return
        if not np.allclose(g.astype(np.float64), w.astype(np.float64), rtol=0, atol=0):
            n = int((g != w).sum())
            bad.append(f"{name}: {n} element(s) differ")

    # per-token and per-atom arrays, straight slices
    eq("asym_id", feat_head["asym_id"], np.asarray(z["ci_feat_asym_id"]).reshape(-1)[kt])
    eq("atom_to_tokatom_idx", feat_head["atom_to_tokatom_idx"],
       np.asarray(z["ci_feat_atom_to_tokatom_idx"]).reshape(-1)[ka])
    eq("head.distogram_rep_atom_mask", feat_head["distogram_rep_atom_mask"],
       np.asarray(z["ci_feat_distogram_rep_atom_mask"]).reshape(-1)[ka])
    eq("head.is_ligand", feat_head["is_ligand"], np.asarray(z["ci_feat_is_ligand"]).reshape(-1)[ka])
    eq("s_inputs", s_inputs, np.asarray(z["ci_s_inputs"])[kt])
    eq("s_trunk", s_trunk, np.asarray(z["ci_s_trunk_f32"])[kt])
    eq("poses", poses, np.asarray(z["ci_coordinate"])[:, ka, :])
    eq("label_coordinate", coords, np.asarray(lab["label_coordinate"])[:, ka, :])
    eq("label_coordinate_mask", cmask, np.asarray(lab["label_coordinate_mask"])[:, ka])
    for k in ("is_rna", "is_dna", "is_ligand", "plddt_m_rep_atom_mask",
              "distogram_rep_atom_mask", "pae_rep_atom_mask"):
        eq(f"loss.{k}", feat_loss[k], np.asarray(lab[f"lf_{k}"]).reshape(-1)[ka])

    # THE PAIR TENSOR, as a window and not merely the right shape
    zt = z_trunk.detach().cpu().view(torch.uint16).numpy()
    want_zt = np.asarray(z["z_trunk"])[np.ix_(kt, kt)]
    if zt.shape != want_zt.shape:
        bad.append(f"z_trunk: shape {zt.shape} vs {want_zt.shape}")
    elif not (zt == want_zt).all():
        bad.append(f"z_trunk: {int((zt != want_zt).sum())} element(s) differ from the k-by-k window")

    # THE REMAPPED INDICES, checked by following them back to the original objects
    a2t_got = feat_head["atom_to_token_idx"].detach().cpu().numpy()
    a2t_want = tok_new[np.asarray(z["ci_feat_atom_to_token_idx"]).reshape(-1)[ka]]
    eq("atom_to_token_idx (remapped)", a2t_got, a2t_want)
    a2t_ok = a2t_got.size == 0 or (a2t_got.min() >= 0 and a2t_got.max() < kt.size)
    if not a2t_ok:
        bad.append(f"atom_to_token_idx out of range for {kt.size} token(s)")
    elif not (kt[a2t_got] == np.asarray(z["ci_feat_atom_to_token_idx"]).reshape(-1)[ka]).all():
        # every kept atom must still report the SAME original token. Only reachable when the indices
        # are in range -- following an out-of-range index would raise instead of reporting.
        bad.append("atom_to_token_idx does not round-trip to the original token ids")

    fai_got = feat_loss["frame_atom_index"].detach().cpu().numpy()
    hf_got = feat_loss["has_frame"].detach().cpu().numpy().astype(bool)
    fai_full = np.asarray(lab["lf_frame_atom_index"])[kt]
    fai_ok = fai_got.size == 0 or (fai_got.min() >= 0 and fai_got.max() < ka.size)
    if not fai_ok:
        bad.append(f"frame_atom_index out of range for {ka.size} atom(s)")
    elif hf_got.any() and not (ka[fai_got[hf_got]] == fai_full[hf_got]).all():
        # where the frame survives, the new indices must point at the SAME original atoms
        bad.append("frame_atom_index does not round-trip to the original atom ids")
    # and a frame may only be marked present if all three of its atoms were kept
    kept_all = np.isin(fai_full, ka).all(axis=-1)
    hf_full = np.asarray(lab["lf_has_frame"]).reshape(-1)[kt].astype(bool)
    if (hf_got & ~kept_all).any():
        bad.append("has_frame is set for a token whose frame atoms were dropped")
    if ((hf_full & kept_all) & ~hf_got).any():
        bad.append("has_frame was cleared for a token whose frame atoms all survived")
    # the head and the loss each carry their own has_frame, from the trunk cache and the label npz.
    # A token framed for one and unframed for the other has no correct reading.
    hf_head = feat_head["has_frame"].detach().cpu().numpy().astype(bool)
    if (hf_head != hf_got).any():
        bad.append(f"has_frame disagrees between head and loss for "
                   f"{int((hf_head != hf_got).sum())} token(s)")
    return bad


def self_test(z, lab, keep_tok, keep_atom, sliced):
    """DOES THE GATE HAVE TEETH? Corrupt each remapping the way a real bug would and require the
    check to fire. A gate that passes on correct code AND on broken code asserts nothing, and the
    two remappings here (atom->token, token->frame atoms) are exactly the ones a reviewer cannot
    verify by eye. Each case mutates a copy and is discarded afterwards."""
    import copy
    out = []
    def mutate(name, fn):
        s2 = list(sliced)
        s2[0] = dict(s2[0]); s2[6] = dict(s2[6])
        fn(s2)
        fired = check_indices(z, lab, keep_tok, keep_atom, tuple(s2), "mutant")
        out.append((name, bool(fired), fired[0] if fired else ""))

    # (1) the classic off-by-one: forget to remap, leave the ORIGINAL token ids in place
    def no_remap_a2t(s2):
        ka = np.nonzero(keep_atom)[0]
        s2[0]["atom_to_token_idx"] = torch.as_tensor(
            np.asarray(z["ci_feat_atom_to_token_idx"]).reshape(-1)[ka],
            device=s2[0]["atom_to_token_idx"].device, dtype=s2[0]["atom_to_token_idx"].dtype)
    mutate("atom_to_token_idx left un-remapped", no_remap_a2t)

    # (2) frames left in the ORIGINAL atom numbering -- scores a pose against another residue's frame
    def no_remap_frames(s2):
        kt = np.nonzero(keep_tok)[0]
        s2[6]["frame_atom_index"] = torch.as_tensor(
            np.asarray(lab["lf_frame_atom_index"])[kt],
            device=s2[6]["frame_atom_index"].device, dtype=s2[6]["frame_atom_index"].dtype)
    mutate("frame_atom_index left un-remapped", no_remap_frames)

    # (3) one atom index shifted by one -- the smallest wrong answer that still looks plausible
    def shift_one_frame(s2):
        f = s2[6]["frame_atom_index"].clone()
        hf = s2[6]["has_frame"].bool()
        idx = int(torch.nonzero(hf)[0]) if hf.any() else 0
        f[idx, 0] = (f[idx, 0] + 1) % s2[0]["atom_to_tokatom_idx"].shape[0]
        s2[6]["frame_atom_index"] = f
    mutate("one frame atom shifted by one", shift_one_frame)

    # (4) has_frame cleared for a token whose frame atoms all survived -- silently drops that
    # residue from the frame-aligned PAE target. Setting has_frame everywhere would NOT be a
    # mutation here, because whole tokens are kept and so no frame breaks in the ordinary path.
    def clear_live_frame(s2):
        hf, hh = s2[6]["has_frame"].clone(), s2[0]["has_frame"].clone()
        live = torch.nonzero(hf.bool())
        if live.numel():
            hf[int(live[0])] = 0; hh[int(live[0])] = 0
        s2[6]["has_frame"], s2[0]["has_frame"] = hf, hh
    mutate("has_frame cleared for a live frame", clear_live_frame)

    # (6) the head's has_frame left un-cleared while the loss's is cleared, or the reverse
    def disagree_has_frame(s2):
        hh = s2[0]["has_frame"].clone()
        hh[0] = 1 - hh[0]
        s2[0]["has_frame"] = hh
    mutate("head and loss has_frame disagree", disagree_has_frame)

    # (5) the pair tensor sliced on rows only -- right shape nowhere, wrong window if it were
    def wrong_z_window(s2):
        kt = np.nonzero(keep_tok)[0]
        s2[3] = torch.from_numpy(np.ascontiguousarray(
            np.asarray(z["z_trunk"])[np.ix_(kt, kt)][::-1].copy())).to(s2[3].device).view(torch.bfloat16)
    mutate("z_trunk window reversed", wrong_z_window)
    return out


def broken_frame_case(z, lab, keep_tok, keep_atom, dev, pdb):
    """THE DEFENSIVE BRANCH, exercised. keep_atom normally holds every atom of every kept token, so
    no frame ever breaks and the clearing code in slice_cached_unit never runs on real data -- the
    'frames cleared' column is 0 everywhere. Drop one frame atom by hand and require that branch to
    clear exactly that token's has_frame and leave the rest of the crop consistent."""
    kt = np.nonzero(keep_tok)[0]
    hf_full = np.asarray(lab["lf_has_frame"]).reshape(-1)[kt].astype(bool)
    fai_full = np.asarray(lab["lf_frame_atom_index"])[kt]
    live = np.nonzero(hf_full)[0]
    if live.size == 0:
        return ["no token with a frame; the broken-frame branch could not be exercised"]
    victim_tok, victim_atom = int(live[0]), int(fai_full[live[0], 0])
    ka2 = keep_atom.copy(); ka2[victim_atom] = False
    sl = BC.slice_cached_unit(z, lab, keep_tok, ka2, dev, torch)
    bad = []
    if sl[8] < 1:
        bad.append(f"dropped atom {victim_atom} of token {kt[victim_tok]} but n_broken = {sl[8]}")
    hf_got = sl[6]["has_frame"].detach().cpu().numpy().astype(bool)
    if hf_got[victim_tok]:
        bad.append("has_frame still set for the token whose frame atom was dropped")
    # every OTHER frame must be untouched: the branch may not clear more than it has to
    others = np.ones(kt.size, bool); others[victim_tok] = False
    kept_all = np.isin(fai_full, np.nonzero(ka2)[0]).all(axis=-1)
    if (hf_got[others] != (hf_full & kept_all)[others]).any():
        bad.append("the broken-frame branch cleared frames it should have left alone")
    bad += check_indices(z, lab, keep_tok, ka2, sl, pdb)
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8, help="structures to test")
    ap.add_argument("--contact", type=float, default=5.0)
    ap.add_argument("--margin", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-model", action="store_true", default=True,
                    help="also run the real head and loss on each crop")
    ap.add_argument("--poses", type=int, default=0,
                    help="poses to run per structure through forward AND backward; 0 = all")
    ap.add_argument("--time", action="store_true",
                    help="also time full-structure vs cropped forward+backward on the same pose")
    ap.add_argument("--out", default="", help="write the per-structure table to this csv")
    ap.add_argument("--no-self-test", action="store_true",
                    help="skip the mutation test that proves the index checks fire")
    a = ap.parse_args()
    if os.environ.get("LAYERNORM_TYPE") != "openfold":
        raise SystemExit("STOP: set LAYERNORM_TYPE=openfold.")

    SKIP = {"9dc3","9bbn","8vba","9q7b","9q5v","9e7x","9b9j","9dzf","9q5m","9mx1"}
    rows = [r for r in csv.DictReader(
        open(os.path.join(ROOT, "train_val_splits/cache_train_structures.csv")))
        if r["tokens"].isdigit() and r["pdb"] not in SKIP]
    rows.sort(key=lambda r: int(r["tokens"]))
    # span the size range deliberately, then fill at random
    picks = [rows[0], rows[len(rows)//2], rows[-1]]
    rest = [r for r in rows if r not in picks]
    random.Random(a.seed).shuffle(rest)
    picks += rest[:max(0, a.n - len(picks))]

    dev = torch.device("cuda:0")
    model, configs = RC.build_model("/home/ebrahim/checkpoint", "protenix-v2", dev)
    configs.train_confidence_only = True
    # THE TRAINER'S WEIGHTS, not Protenix's shipped defaults. At the shipped alpha_confidence = 1e-4
    # and alpha_pae = 0.0 the cumulative loss comes out around 5e-4 and the PAE term -- the one that
    # ipTM is derived from -- is multiplied by zero, so a gate run that way would report a number
    # unrelated to what training optimises.
    configs.loss.weight.alpha_pae = 1.0
    configs.loss.weight.alpha_confidence = 1.0
    # TRAINING MODE, as the trainer runs it: Protenix offloads pae/pde to the CPU above
    # PROTENIX_CONF_OFFLOAD_TOKENS (2000) only when the module is NOT training, and the loss would
    # then compute on the CPU. Matching the trainer keeps this gate honest about where tensors live.
    model.confidence_head.train()
    # GRADIENTS ON THE HEAD ONLY, as FT3 trains it. Without this every backward below would be a
    # no-op and the "backward produced no gradient" check would fire on every pose.
    for prm in model.parameters():
        prm.requires_grad_(False)
    for prm in model.confidence_head.parameters():
        prm.requires_grad_(True)
    loss_fn = ProtenixLoss(configs)

    print(f"contact {a.contact:.0f} A, margin {a.margin:.0f} A, {len(picks)} structure(s)\n")
    print(f"{'structure':10s} {'tokens':>7s} {'kept':>7s} {'%':>6s} {'frames cleared':>14s} "
          f"{'union-1':>6s} {'index checks':>13s} {'loss on crop':>13s}")
    failures, fracs, fulls, table, timing = [], [], [], [], []
    for r in picks:
        pdb = r["pdb"]
        tp = lp = ""
        for root in ROOTS:
            for seed in SEEDS:
                c_t = os.path.join(ROOT, root, "trunks", pdb, f"{seed}.npz")
                c_l = os.path.join(ROOT, root, "labels", pdb, f"{seed}.npz")
                if os.path.exists(c_t) and os.path.exists(c_l):
                    tp, lp = c_t, c_l; break
            if tp: break
        if not tp:
            print(f"{pdb:10s} {'-':>7s} no trunk+label pair in any fold root; skipped"); continue
        z, lab = np.load(tp), np.load(lp, allow_pickle=True)
        keep_tok, keep_atom = BC.interface_crop_tokens(z, lab, a.contact, a.margin)
        n_tok = np.asarray(z["ci_feat_asym_id"]).reshape(-1).shape[0]
        if keep_tok is None:
            print(f"{pdb:10s} {n_tok:7d} {'n/a':>7s} {'-':>6s} {'-':>14s} "
                  f"{'NO CROP':>13s} {'(full)':>13s}")
            continue
        sliced = BC.slice_cached_unit(z, lab, keep_tok, keep_atom, dev, torch)
        bad = check_indices(z, lab, keep_tok, keep_atom, sliced, pdb)
        k = int(keep_tok.sum()); fracs.append(k / n_tok); fulls.append(n_tok)
        lossinfo, terms = "not run", ""
        if a.run_model:
            fh, s_in, s_tr, zt, poses, _nc, fl, (co, cm), nb = sliced
            n_pose = poses.shape[0] if a.poses <= 0 else min(a.poses, poses.shape[0])
            try:
                cums, per_term = [], {t: [] for t in TERMS}
                for pi in range(n_pose):
                    # FORWARD AND BACKWARD, as training runs it. no_grad was the gap review round 5
                    # named: a sliced tensor that is a non-contiguous view, or a remapped index used
                    # as a gather source, can forward cleanly and fail or silently zero in backward.
                    cum, vals = run_one_pose(model, configs, loss_fn, fh, s_in, s_tr, zt,
                                             poses, fl, co, cm, pi, dev, backward=True)
                    cums.append(cum)
                    for t in TERMS:
                        per_term[t].append(vals[t])
                    if not np.isfinite(cum) or not all(np.isfinite(v) for v in vals.values()):
                        bad.append(f"pose {pi}: non-finite loss {vals}")
                    g = [n for n, prm in model.confidence_head.named_parameters()
                         if prm.grad is not None and not torch.isfinite(prm.grad).all()]
                    if g:
                        bad.append(f"pose {pi}: non-finite gradient in {len(g)} tensor(s), "
                                   f"first {g[0]}")
                    if all(prm.grad is None or float(prm.grad.abs().sum()) == 0.0
                           for _n, prm in model.confidence_head.named_parameters()):
                        bad.append(f"pose {pi}: backward produced no gradient at all")
                    model.zero_grad(set_to_none=True)
                vals = {t: float(np.mean(per_term[t])) for t in TERMS}
                lossinfo = f"{np.mean(cums):.4f}"
                terms = " ".join(f"{t.split('_')[0]} {vals[t]:.3f}" for t in TERMS)
                terms += f"  [{n_pose} pose(s), fwd+bwd]"
            except torch.cuda.OutOfMemoryError as e:
                bad.append(f"OOM: {str(e)[:90]}")
                lossinfo = "OOM"
            except Exception as e:
                bad.append(f"{type(e).__name__}: {e}")
                lossinfo = "RAISED"
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        if a.time:
            # SAME POSE, SAME CODE PATH, full structure versus crop. The full build goes through
            # slice_cached_unit with all-true masks so the only difference being timed is the size.
            try:
                full_t = time_fwd_bwd(
                    model, configs, loss_fn,
                    lambda: BC.slice_cached_unit(z, lab, np.ones(n_tok, bool),
                                                 np.ones(keep_atom.shape[0], bool), dev, torch), dev)
                crop_t = time_fwd_bwd(
                    model, configs, loss_fn,
                    lambda: BC.slice_cached_unit(z, lab, keep_tok, keep_atom, dev, torch), dev)
                timing.append((pdb, n_tok, k, full_t, crop_t))
            except torch.cuda.OutOfMemoryError:
                timing.append((pdb, n_tok, k, float("nan"), float("nan")))
                torch.cuda.empty_cache()

        # THE UNION IS NOT A GUESS: assert no single pose needs a token the union dropped.
        try:
            extra = BC.assert_pose_invariant_crop(z, lab, a.contact, a.margin)
        except Exception as e:
            bad.append(f"pose-invariance: {type(e).__name__}: {e}")
            extra = -1

        nb = sliced[8]
        print(f"{pdb:10s} {n_tok:7d} {k:7d} {k/n_tok*100:5.1f}% {nb:14d} {extra:6d} "
              f"{('OK' if not bad else f'{len(bad)} FAIL'):>13s} {lossinfo:>13s}   {terms}")
        row = {"pdb": pdb, "trunk": os.path.relpath(tp, ROOT), "tokens": n_tok,
               "kept_tokens": k, "kept_frac": round(k / n_tok, 4), "frames_cleared": nb,
               "index_checks": "OK" if not bad else "FAIL", "loss": lossinfo}
        if lossinfo not in ("RAISED", "not run"):
            row.update({t: round(vals[t], 6) for t in TERMS})
        table.append(row)
        for b in bad:
            print(f"           ! {b}")
            failures.append((pdb, b))

    # THE GATE'S OWN GATE
    if not a.no_self_test:
        r0 = picks[0]
        tp = lp = ""
        for root in ROOTS:
            for seed in SEEDS:
                c_t = os.path.join(ROOT, root, "trunks", r0["pdb"], f"{seed}.npz")
                c_l = os.path.join(ROOT, root, "labels", r0["pdb"], f"{seed}.npz")
                if os.path.exists(c_t) and os.path.exists(c_l):
                    tp, lp = c_t, c_l; break
            if tp: break
        z, lab = np.load(tp), np.load(lp, allow_pickle=True)
        kt_m, ka_m = BC.interface_crop_tokens(z, lab, a.contact, a.margin)
        print(f"\nmutation test on {r0['pdb']} -- each row is a bug the checks must catch")
        sl = BC.slice_cached_unit(z, lab, kt_m, ka_m, dev, torch)
        for name, fired, first in self_test(z, lab, kt_m, ka_m, sl):
            print(f"  {('CAUGHT' if fired else 'MISSED'):>7s}  {name:42s} {first}")
            if not fired:
                failures.append((r0["pdb"], f"mutation NOT caught: {name}"))
        bf = broken_frame_case(z, lab, kt_m, ka_m, dev, r0["pdb"])
        print(f"  {('OK' if not bf else 'FAIL'):>7s}  "
              f"{'one frame atom dropped by hand':42s} "
              f"{'has_frame cleared for exactly that token' if not bf else bf[0]}")
        for b in bf:
            failures.append((r0["pdb"], f"broken-frame branch: {b}"))

    if fracs:
        fr, fu = np.array(fracs), np.array(fulls, dtype=float)
        sp = (fu ** 2.21).sum() / ((fu * fr) ** 2.21).sum()
        print(f"\nretained: median {np.median(fr)*100:.1f}%  mean {fr.mean()*100:.1f}%"
              f"  | projected speedup {sp:.2f}x at the measured tokens^2.21")
    if timing:
        ok_t = [t for t in timing if np.isfinite(t[3]) and np.isfinite(t[4])]
        print(f"\nMEASURED forward+backward, one pose, median of 3 (card {os.environ.get('CUDA_VISIBLE_DEVICES','?')})")
        print(f"  {'structure':10s} {'tokens':>7s} {'kept':>7s} {'full s':>8s} {'crop s':>8s} {'x':>6s}")
        for pdb_, nt_, k_, ft_, ct_ in ok_t:
            print(f"  {pdb_:10s} {nt_:7d} {k_:7d} {ft_:8.3f} {ct_:8.3f} {ft_/ct_:6.2f}")
        if ok_t:
            tf = sum(t[3] for t in ok_t); tc = sum(t[4] for t in ok_t)
            print(f"  pooled over {len(ok_t)} structure(s): {tf:.2f} s full, {tc:.2f} s cropped, "
                  f"{tf/tc:.2f}x")
        n_oom = len(timing) - len(ok_t)
        if n_oom:
            print(f"  {n_oom} structure(s) could not be timed at full size on this card (OOM)")

    if a.out and table:
        keys = list(table[0].keys())
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for row in table:
                w.writerow({kk: row.get(kk, "") for kk in keys})
        print(f"wrote {a.out} ({len(table)} row(s))")
    if failures:
        print(f"\nGATE FAILED: {len(failures)} problem(s).")
        return 1
    print("\nGATE PASSED: every sliced array matches the full array indexed by the kept mask, the "
          "remapped indices round-trip to the original tokens and atoms, no frame survives without "
          "its atoms, and the head and loss accept every crop with finite values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
