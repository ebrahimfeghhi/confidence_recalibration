#!/usr/bin/env python
"""Exact DockQ between every pair of poses of one interface, vectorised on the GPU.

SUMMARY
    Computes the full N x N DockQ matrix for one interface, where every pose is the same complex (same
    sequence, same residues), so residue correspondence is exact and no alignment is needed. That is
    what makes it batchable; the general DockQ implementation cannot assume it.

    NOT AN APPROXIMATION. It reproduces DockQ 2.1.3:

        contact     residue pair within 5.0 A, the MIN over all their atom pairs
        interface   residues within 10.0 A of the partner, taken from the REFERENCE pose
        backbone    CA, C, N, O
        fnat        |contacts(ref) & contacts(model)| / |contacts(ref)|
        iRMSD       RMSD after superposing the model's interface backbone onto the reference's
        LRMSD       superpose on the RECEPTOR (the chain with more residues), then RMSD of the ligand
                    backbone -- no second superposition
        DockQ       (fnat + 1/(1+(iRMSD/1.5)^2) + 1/(1+(LRMSD/8.5)^2)) / 3

    Why each term batches: fnat's contact masks are per-pose booleans, so all-pairs intersection is one
    matmul; LRMSD's receptor and ligand atoms are the same in every pose, so one batched Kabsch covers
    all pairs; iRMSD's interface depends on the reference, so references loop and models batch. The
    reference package takes 13.8 ms per pair (~100 core-hours at N=5125); this takes minutes.

    A pose with no interface contacts is undefined as a REFERENCE (DockQ returns None there); its row
    is set to 0, matching the reference pipeline.

    Validated against OST's DockQ, an independent implementation, when written. The ESM-era driver and
    that validation (build(), validate_against_ground_truth()) are in git history before 2026-09-25;
    this copy keeps only what pose_to_pose_dockq.py uses.

INPUTS
    none -- a library; dockq_matrix() takes [(pose path, chain 1, chain 2)], ONE chain pair per pose

OUTPUTS
    none -- dockq_matrix() returns the matrix

EXAMPLE
    from dockq_torch import dockq_matrix
    M = dockq_matrix([(p, ab, ag) for p, ab, ag in items], device="cuda")   # M[i, j]: ref i, model j

FUNCTIONS
    load_poses():     N poses -> all-atom and backbone coordinate tensors, one chain pair per pose
    residue_sqdist(): coordinates -> squared min distance between every residue pair, per pose
    kabsch_rms():     batched superposition -> RMSD
    kabsch_rt():      batched superposition -> rotation and translation
    dockq_matrix():   N poses -> the N x N DockQ matrix
"""
import time
from pathlib import Path
import numpy as np
import torch

BACKBONE = ("CA", "C", "N", "O")
FNAT_T2 = 5.0 ** 2          # DockQ works in SQUARED distances
IFACE_T2 = 10.0 ** 2


# ---------------------------------------------------------------- loading
def load_poses(items, device, skip=()):
    """in: [(pose path, chain 1, chain 2)] of ONE complex, a device, atom names to leave out (e.g.
    ("OXT",) to match OST's clean-up; default none, as the pipeline has always scored).
    out: (all1, all2, bb1, bb2, seg1, seg2) tensors, described below

    ONE CHAIN PAIR PER POSE, because the model chain letters are NOT fixed.

    The scoring CSV names chains by their NATIVE letters; the model files use different letters,
    and the `chain_mapping` column (ref -> model) translates. On multi-copy assemblies the model
    builds two copies of everything and OST maps the native onto whichever copy fits, so the
    mapping CHANGES FROM POSE TO POSE (e.g. 8oni_AE uses model chains A-E, A-F, B-E or B-F
    depending on the seed). Using the native letters directly on the model file scores the wrong
    pair of molecules -- which is exactly the bug this signature exists to prevent.

    all1/all2   : [N, nat_k, 3]  every atom, for residue-residue contact distances
    bb1/bb2     : [N, nres_k, 4, 3]  backbone CA,C,N,O, for the RMSDs
    seg1/seg2   : [nat_k] residue index of each atom (to take a MIN per residue pair)
    """
    import gemmi
    A1 = A2 = B1 = B2 = None
    seg1 = seg2 = None
    N = len(items)
    for n, (p, ch1, ch2) in enumerate(items):
        st = gemmi.read_structure(str(p))
        st.setup_entities()
        m = st[0]
        out = []
        for ch in (ch1, ch2):
            chain = m[ch]
            all_xyz, bb_xyz, seg = [], [], []
            for ri, res in enumerate(chain):
                bb = {}
                for at in res:
                    if at.name in skip:
                        continue
                    all_xyz.append([at.pos.x, at.pos.y, at.pos.z]); seg.append(ri)
                    if at.name in BACKBONE:
                        bb[at.name] = [at.pos.x, at.pos.y, at.pos.z]
                # a residue missing a backbone atom would break the fixed shape; use CA as filler
                ca = bb.get("CA", [0.0, 0.0, 0.0])
                bb_xyz.append([bb.get(a, ca) for a in BACKBONE])
            out.append((np.asarray(all_xyz, np.float32), np.asarray(bb_xyz, np.float32),
                        np.asarray(seg, np.int64)))
        (a1, b1, s1), (a2, b2, s2) = out
        if A1 is None:
            A1 = torch.empty((N,) + a1.shape, dtype=torch.float32)
            A2 = torch.empty((N,) + a2.shape, dtype=torch.float32)
            B1 = torch.empty((N,) + b1.shape, dtype=torch.float32)
            B2 = torch.empty((N,) + b2.shape, dtype=torch.float32)
            seg1 = torch.from_numpy(s1); seg2 = torch.from_numpy(s2)
        # Every pose must yield the SAME shapes. Different poses may use different CHAIN LETTERS
        # (multi-copy assemblies: OST maps the native onto whichever model copy fits), but the
        # chains are copies of the same molecules, so residue and atom counts must agree. If they
        # do not, the chain mapping is wrong and we would be comparing different molecules.
        if a1.shape != A1.shape[1:] or a2.shape != A2.shape[1:]:
            raise ValueError(
                f"pose {n} ({Path(p).name}, chains {ch1}/{ch2}) has shapes "
                f"{a1.shape}/{a2.shape}, expected {tuple(A1.shape[1:])}/{tuple(A2.shape[1:])} — "
                f"the chain mapping is inconsistent")
        A1[n] = torch.from_numpy(a1); B1[n] = torch.from_numpy(b1)
        A2[n] = torch.from_numpy(a2); B2[n] = torch.from_numpy(b2)
    return (A1.to(device), A2.to(device), B1.to(device), B2.to(device),
            seg1.to(device), seg2.to(device))


# ---------------------------------------------------------------- residue distances
def residue_sqdist(all1, all2, seg1, seg2, n1, n2, chunk=64):
    """in: all-atom coordinates and residue indices of both chains, residue counts, a chunk size.
    out: [N, n1, n2] SQUARED min distance between every residue pair, per pose

    DockQ's residue_distances(): min over all atom-atom pairs of the two residues.
    """
    N = all1.shape[0]
    out = torch.empty((N, n1, n2), dtype=torch.float32, device=all1.device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d = torch.cdist(all1[s:e], all2[s:e]).pow_(2)              # [b, nat1, nat2]
        b = d.shape[0]
        # segment-min down atoms of chain1, then across atoms of chain2
        t = torch.full((b, n1, d.shape[2]), float("inf"), device=d.device)
        t.scatter_reduce_(1, seg1.view(1, -1, 1).expand(b, -1, d.shape[2]), d,
                          reduce="amin", include_self=True)
        u = torch.full((b, n1, n2), float("inf"), device=d.device)
        u.scatter_reduce_(2, seg2.view(1, 1, -1).expand(b, n1, -1), t,
                          reduce="amin", include_self=True)
        out[s:e] = u
    return out


# ---------------------------------------------------------------- Kabsch
def kabsch_rms(P, Q):
    """in: P, Q = [B, m, 3].  out: [B] RMSD after superposing P onto Q (Biopython's SVDSuperimposer)"""
    Pc = P - P.mean(1, keepdim=True)
    Qc = Q - Q.mean(1, keepdim=True)
    H = Pc.transpose(1, 2) @ Qc
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.eye(3, device=P.device).expand(P.shape[0], 3, 3).clone()
    D[:, 2, 2] = d
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    Pr = Pc @ R.transpose(1, 2)
    return (Pr - Qc).pow(2).sum(-1).mean(-1).sqrt()


def kabsch_rt(P, Q):
    """in: P, Q = [B, m, 3].  out: (R, t) with P @ R.T + t ~ Q, per batch member"""
    pm = P.mean(1, keepdim=True); qm = Q.mean(1, keepdim=True)
    H = (P - pm).transpose(1, 2) @ (Q - qm)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.eye(3, device=P.device).expand(P.shape[0], 3, 3).clone()
    D[:, 2, 2] = d
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    t = qm - pm @ R.transpose(1, 2)
    return R, t


# ---------------------------------------------------------------- the matrix
def dockq_matrix(items, device="cuda", verbose=True, timings=None, skip=()):
    """in: [(pose path, chain 1, chain 2)], a device, verbosity, an optional dict for timings, atom
    names to leave out (see load_poses).
    out: the [N, N] DockQ matrix as numpy; entry [i, j] = DockQ(reference = pose i, model = pose j)

    One chain pair PER POSE (see load_poses). Pass timings={} to get load and compute seconds back.
    """
    t0 = time.time()
    all1, all2, bb1, bb2, seg1, seg2 = load_poses(items, device, skip)
    t_load = time.time() - t0
    if timings is not None:
        timings["load_s"] = t_load
    N = all1.shape[0]; n1 = bb1.shape[1]; n2 = bb2.shape[1]
    if verbose:
        print(f"  loaded {N} poses | {n1} + {n2} residues | {t_load:.0f}s", flush=True)

    D = residue_sqdist(all1, all2, seg1, seg2, n1, n2)          # [N, n1, n2] squared
    del all1, all2
    torch.cuda.empty_cache() if str(device).startswith("cuda") else None

    # ---- fnat: one matmul over flattened contact masks ----
    C = (D < FNAT_T2).reshape(N, -1).float()                    # [N, n1*n2]
    nat = C.sum(1)                                              # contacts of each pose as REF
    shared = C @ C.T                                            # [N, N] shared contacts
    fnat = shared / nat.clamp(min=1).unsqueeze(1)               # row i = reference i
    del C, shared

    # A pose with NO interface contacts is a completely failed dock -- the antibody never touches
    # the antigen. DockQ is UNDEFINED there (calc_DockQ: `if nat_total == 0: return None`), and
    # the reference pipeline stores 0.0. Such a pose is only invalid as the REFERENCE, so this
    # zeroes its ROW. Without this we compute a spurious nonzero DockQ (the iRMS/LRMS terms
    # survive, giving errors up to 2/3) or NaN (Kabsch on an empty interface).
    no_iface = nat == 0
    if verbose and no_iface.any():
        print(f"  {int(no_iface.sum())}/{N} poses have NO interface contacts -> "
              f"DockQ undefined as reference, row set to 0", flush=True)
    if verbose: print(f"  fnat done | {time.time()-t0:.0f}s", flush=True)

    # ---- LRMSD: receptor/ligand subsets are FIXED -> batch over all pairs ----
    recep, lig = (bb1, bb2) if n1 > n2 else (bb2, bb1)          # receptor = MORE residues
    R_flat = recep.reshape(N, -1, 3); L_flat = lig.reshape(N, -1, 3)
    lrms = torch.empty((N, N), device=device)
    for i in range(N):
        Rr, tt = kabsch_rt(R_flat, R_flat[i].expand(N, -1, -1))   # map each model onto ref i
        Lr = L_flat @ Rr.transpose(1, 2) + tt
        lrms[i] = (Lr - L_flat[i]).pow(2).sum(-1).mean(-1).sqrt()
        if verbose and i % 512 == 0:
            print(f"  lrms {i}/{N} | {time.time()-t0:.0f}s", flush=True)

    # ---- iRMSD: interface subset comes from the REFERENCE -> loop refs, batch models ----
    irms = torch.zeros((N, N), device=device)
    for i in range(N):
        if no_iface[i]:
            continue                                            # undefined as reference; row stays 0
        m = D[i] < IFACE_T2
        i1 = m.any(1); i2 = m.any(0)                            # interface residues, from ref i
        ref = torch.cat([bb1[i][i1].reshape(-1, 3), bb2[i][i2].reshape(-1, 3)], 0)
        mod = torch.cat([bb1[:, i1].reshape(N, -1, 3), bb2[:, i2].reshape(N, -1, 3)], 1)
        irms[i] = kabsch_rms(mod, ref.expand(N, -1, -1))
        if verbose and i % 512 == 0:
            print(f"  irms {i}/{N} | {time.time()-t0:.0f}s", flush=True)

    dq = (fnat + 1 / (1 + (irms / 1.5) ** 2) + 1 / (1 + (lrms / 8.5) ** 2)) / 3
    dq[no_iface] = 0.0                                          # DockQ undefined for that reference
    if timings is not None:
        timings["total_s"] = time.time() - t0
        timings["compute_s"] = timings["total_s"] - timings["load_s"]
        timings["n_no_iface"] = int(no_iface.sum())
    if verbose: print(f"  DONE {N}x{N} in {time.time()-t0:.0f}s", flush=True)
    return dq.cpu().numpy()
