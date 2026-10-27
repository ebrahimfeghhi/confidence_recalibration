#!/usr/bin/env python
"""Poses grouped by INTERFACE, batched as [B, K] because two of the three losses need siblings.

THE EXAMPLE IS AN INTERFACE, NOT A STRUCTURE. 382 of our 487 training structures carry more than
one labelled antibody-antigen interface (311 have two, 40 three, 29 four, 2 six), so grouping by
pdb would merge distinct interfaces -- with distinct DockQ labels -- into one ranking group. The
key here is `7zra_AC`, exactly as results.csv names it.

AND THE BATCH IS [B, K]. The ranking loss correlates predictions against truth across a complex's
poses, and the distance-preservation loss compares pairwise gaps among them; both are undefined on
a single pose. So each item is one interface with K of its poses, sampled per epoch out of the 50
available -- which is upstream's samples_per_complex, and which also means a pose combination is
rarely repeated.

WHICH CHAINS AN INTERFACE IS depends on the POSE: phase_k_eval resolves it per pose through OST
because "the first antigen is not determined by sequence", and OST disagreed with the static
chain_roles table on 145 poses across 2 structures -- one of them 7zra. This dataset therefore
reads the chain pair from the dump's own asym_i/asym_j rather than assuming a letter order.
"""
import collections
import csv
import json
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import features as FT


class InterfacePoses(Dataset):
    def __init__(self, pool_csv, dump_dir, restype_dir, results_csv, k=10, max_n=FT.MAX_N,
                 seed=0, require_full=True):
        self.dump, self.rt, self.k, self.max_n = dump_dir, restype_dir, k, max_n
        self.rng = np.random.default_rng(seed)
        self.labels = self._labels(results_csv)
        self.items = self._index(pool_csv, require_full)
        self._cache = {}

    @staticmethod
    def _labels(results_csv):
        """(iface, seed, sample) -> DockQ. The label join is by identity, never by position."""
        out = {}
        with open(results_csv) as fh:
            for r in csv.DictReader(fh):
                try:
                    out[(r["iface"], int(r["seed"]), int(r["sample"]))] = float(r["dockq"])
                except (KeyError, ValueError):
                    continue
        return out

    def _index(self, pool_csv, require_full):
        """One entry per interface that has poses on disk AND labels for them."""
        want = collections.defaultdict(list)
        with open(pool_csv) as fh:
            for r in csv.DictReader(fh):
                want[r["pdb"]].append((r["chain1"], r["chain2"]))
        items = []
        for pdb, chains in sorted(want.items()):
            poses = sorted(glob.glob(os.path.join(self.dump, pdb, "*.npz")))
            poses = [p for p in poses if "_paesum" not in os.path.basename(p)
                     and "_pae_mean" not in os.path.basename(p)]
            if not poses:
                continue
            if not os.path.exists(os.path.join(self.dump, pdb, "_pae_mean.npz")):
                # WITHOUT THE DENSE MEAN EVERY CENTRED FEATURE IS WRONG, and wrong in a direction
                # that flatters every pose. Skipping is the only safe default.
                continue
            for c1, c2 in chains:
                iface = f"{pdb}_{c1}{c2}"
                keyed = [(p, *self._key(p)) for p in poses]
                # ELIGIBILITY IS DECIDED HERE, NOT AFTER SAMPLING. A pose needs a label AND an
                # OST chain mapping for this interface; dropping one after the group of K was
                # drawn left x with K slots while lengths and y had K-1, so collate or the model
                # saw inconsistent shapes -- and if every drawn pose were unresolved, max(lens)
                # raised before the item could be returned. Caught in review.
                cm = self._chainmap_safe(pdb)
                ok = [(p, s, k) for p, s, k in keyed
                      if (iface, s, k) in self.labels
                      and isinstance(cm.get(f"{s}_{k}"), dict)
                      and iface in cm.get(f"{s}_{k}", {})]
                if require_full and len(ok) < self.k:
                    continue
                if ok:
                    items.append({"iface": iface, "pdb": pdb, "poses": ok})
        return items

    @staticmethod
    def _key(path):
        """(seed, sample) from <seed>_<sample>.npz -- the names the dump wrote from the
        coordinate-identity join, not a positional guess."""
        stem = os.path.basename(path)[:-len(".npz")]
        s, k = stem.split("_")
        return int(s), int(k)

    def _complex(self, pdb, n_tokens):
        if pdb not in self._cache:
            m = np.load(os.path.join(self.dump, pdb, "_pae_mean.npz"))["pae_col_mean"]
            letters = np.load(os.path.join(self.rt, f"{pdb}.npz"))["residue_one_letter"]
            if len(letters) != m.shape[0]:
                raise SystemExit(f"STOP: {pdb}: {len(letters)} residue letter(s) against a "
                                 f"{m.shape} PAE mean. Not the same molecule.")
            self._cache[pdb] = (m, letters)
        return self._cache[pdb]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        it = self.items[n]
        poses = it["poses"]
        pick = (self.rng.choice(len(poses), self.k, replace=len(poses) < self.k)
                if len(poses) != self.k else np.arange(self.k))
        xs, lens, ys = [], [], []
        mean = letters = None
        for p in pick:
            path, sd, smp = poses[int(p)]
            z = np.load(path)
            if mean is None:
                mean, letters = self._complex(it["pdb"], int(z["n_tokens"]))
            ch = self._chains(z, it["iface"], it["pdb"], sd, smp)
            if ch is None:
                # Cannot happen: _index() only admits poses whose mapping resolves. Raising
                # rather than skipping, because silently returning K-1 poses is the bug this
                # eligibility filter exists to remove.
                raise SystemExit(f"STOP: {it['iface']} {sd}_{smp} lost its chain mapping between "
                                 f"indexing and sampling.")
            f, npair = FT.build(z, mean, letters, keep_chains=ch, max_n=self.max_n)
            xs.append(f)
            lens.append(npair)
            ys.append(FT.logit_label(self.labels[(it["iface"], sd, smp)]))
        N = max(1, max(lens))
        x = np.zeros((self.k, N, FT.FEATURE_DIM), dtype=np.float32)
        for i, f in enumerate(xs):
            if len(f):
                x[i, :len(f)] = f
        return {"x": torch.from_numpy(x), "lengths": torch.tensor(lens, dtype=torch.long),
                "y": torch.tensor(ys, dtype=torch.float32), "iface": it["iface"]}

    def _chainmap_safe(self, pdb):
        """The chain map, or {} when absent -- used at INDEX time, where a missing map should
        exclude the complex rather than abort the whole dataset build."""
        try:
            return self._chainmap(pdb)
        except SystemExit:
            return {}

    def _chainmap(self, pdb):
        """OST's per-pose interface -> (asym_i, asym_j), from abag_rank/chain_map.py."""
        key = ("cm", pdb)
        if key not in self._cache:
            f = os.path.join(self.dump, pdb, "_chainmap.json")
            if not os.path.exists(f):
                raise SystemExit(
                    f"STOP: {pdb}: no _chainmap.json. An interface is a CHAIN PAIR and which asym "
                    f"pair it is gets decided PER POSE by OST -- inferring it from the interface's "
                    f"letters is the assumption that mismatched ipTM and DockQ on 7zra. Run "
                    f"abag_rank/chain_map.py over this dump first.")
            with open(f) as fh:
                self._cache[key] = json.load(fh)
        return self._cache[key]

    def _chains(self, z, iface, pdb, seed, sample):
        """The (asym_i, asym_j) OST assigned THIS pose for THIS interface. Never inferred."""
        cm = self._chainmap(pdb).get(f"{seed}_{sample}")
        if not cm or "error" in cm:
            return None
        pair = cm.get(iface)
        return (int(pair[0]), int(pair[1])) if pair else None
