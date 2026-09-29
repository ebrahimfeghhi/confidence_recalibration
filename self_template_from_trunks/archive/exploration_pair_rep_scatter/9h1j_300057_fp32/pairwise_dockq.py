"""Pairwise pose-to-pose DockQ between the 5 samples of this fold, for each of the 4 nanobody-peptide chain pairs.
Chain names are fixed by input order (A-D nanobody, E-H peptide), so pair (A,E) is the same pair in every sample.
Uses dockq_torch.dockq_matrix (exact DockQ 2.1.3 reimplementation), symmetrised as in pose_to_pose_dockq.py."""
import sys, glob, json, numpy as np
sys.path.insert(0, '/home/ebrahim/tts_abag/diffusion_commit_clean_v2')
from dockq_torch import dockq_matrix
P = sorted(glob.glob('poses/9h1j/*/predictions/9h1j_sample_?.cif')); assert len(P) == 5
res = {}
for a, b in [('A','E'),('B','F'),('C','G'),('D','H')]:
    M = np.asarray(dockq_matrix([(p, a, b) for p in P], device='cuda', verbose=False)); S = (M + M.T) / 2
    iu = np.triu_indices(5, 1); v = S[iu]
    print(f'pair {a}{b}: pairwise DockQ mean {v.mean():.3f} min {v.min():.3f} max {v.max():.3f}; std of pairwise {v.std():.3f}')
    print(np.round(S, 2))
    res[a+b] = S.tolist()
json.dump(res, open('pairwise_dockq.json', 'w'))
