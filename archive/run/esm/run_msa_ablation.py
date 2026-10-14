"""run_msa_ablation.py — the MSA ablation: base ESMFold2 (MSA-enabled) vs ESMFold2-Fast (single-seq).

THE THREAT THIS SETTLES.
docs/confidence_head_hypothesis.md argues Protenix out-selects ESMFold2 because its confidence
head is a rollout-trained ERROR DISCRIMINATOR. But ESMFold2-Fast runs SINGLE-SEQUENCE --
`msa_encoder.enabled = false` in that checkpoint, and run_foldbench_abag.py never passes an MSA
(it even filters MSA-bearing JSONs out at :46-48). Protenix uses MSAs. So the entire comparison
might be MSA vs NO-MSA, and every conclusion in that document would be confounded.

msa_confound.py showed Protenix's advantage does NOT scale with MSA DEPTH (Spearman -0.299 -- it
runs backwards). But depth there never drops below 10,262 sequences, so it compares deep against
deeper and says nothing about 0 vs 10,262 -- which is the actual contrast. Only this experiment
settles it.

WHAT IT DOES.
Runs base `biohub/ESMFold2` (msa_encoder.enabled = TRUE -- verified in the cached config) with
Protenix's OWN MSAs, on the 14 complexes of the selection-critical band, at the same depth
(50 seeds x 5 diffusion samples) and the same TTS settings (10% LM mask, 30% dropout) as the
single-sequence run. Everything else is held fixed, so the only variable is the MSA.

READ:  Protenix's `-updated` JSONs, whose proteinChain entries carry `pairedMsaPath` /
       `unpairedMsaPath` -> a3m files under Protenix/foldbench_abag/prep/<cx>/msa/<n>/.
WRITE: the same layout the eval expects, so eval/eval_ost_foldbench.py scores it unchanged:
       <out>/<complex>/seed_<N>/predictions/{<cx>_sample_<k>.cif, <cx>_summary_confidence_sample_<k>.json}

INTERPRETATION.
  ESM-with-MSA selector accuracy jumps to Protenix's level  -> the confidence-head story is DEAD;
                                                               it was MSAs all along.
  It stays at the single-sequence level                     -> MSAs are NOT the explanation, and
                                                               the confidence-head account stands.
"""

import argparse
import json
import os
import string
import sys
import time
from pathlib import Path

import numpy as np
import torch

from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
from esm.utils.msa.msa import MSA
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import JSON_DIR                                        # noqa: E402

CHAIN_LETTERS = string.ascii_uppercase


def load_msa(a3m_path, max_depth):
    """a3m -> ESM MSA, capped at max_depth rows (row 0 is the query)."""
    if not a3m_path or not os.path.exists(a3m_path):
        return None
    try:
        msa = MSA.from_a3m(a3m_path)
    except Exception as e:
        print(f"    ! MSA.from_a3m failed on {a3m_path}: {e}", flush=True)
        return None
    if len(msa.entries) > max_depth:
        msa = MSA(entries=msa.entries[:max_depth])
    return msa


def load_complexes(names, max_depth, no_msa=False):
    """Protenix '-updated' JSONs (the ones WITH MSA paths -- the ESM driver deliberately skips
    these) -> [(name, StructurePredictionInput)] with an MSA attached to every protein chain."""
    out = []
    for name in names:
        cands = sorted(Path(JSON_DIR).glob(f"{name}*updated*.json")) or \
                sorted(Path(JSON_DIR).glob(f"{name}*.json"))
        if not cands:
            print(f"  ! no JSON for {name}", flush=True)
            continue
        d = json.loads(cands[0].read_text())
        d = d[0] if isinstance(d, list) else d
        inputs, chain_idx, depths = [], 0, []
        for entry in d["sequences"]:
            if "proteinChain" not in entry:
                continue                                   # ligands/ions: ESMFold2 is protein-only
            cd = entry["proteinChain"]
            seq = cd["sequence"]
            count = cd.get("count", 1)
            ids = [CHAIN_LETTERS[chain_idx + i] for i in range(count)]
            chain_idx += count
            # unpaired MSA is the per-chain homolog set -- the coevolutionary signal we care about.
            # The CONTROL arm attaches none, so the same checkpoint runs single-sequence.
            msa = None if no_msa else load_msa(cd.get("unpairedMsaPath"), max_depth)
            depths.append(len(msa.entries) if msa else 0)
            inputs.append(ProteinInput(id=ids if count > 1 else ids[0], sequence=seq, msa=msa))
        if not inputs:
            continue
        print(f"  {name}: {len(inputs)} chains, MSA depths {depths}", flush=True)
        out.append((name, StructurePredictionInput(sequences=inputs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--complexes", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--model", default="biohub/ESMFold2",
                    help="base ESMFold2 has msa_encoder.enabled=true; -Fast has it false")
    ap.add_argument("--lm_mask_pct", type=float, default=0.1)      # same TTS settings as the
    ap.add_argument("--lm_dropout", type=float, default=0.3)       # single-sequence run
    ap.add_argument("--num_loops", type=int, default=16)
    ap.add_argument("--num_sampling_steps", type=int, default=16)
    ap.add_argument("--num_diffusion_samples", type=int, default=5)
    ap.add_argument("--msa_max_depth", type=int, default=1024)     # processor.py default
    ap.add_argument("--chunk_size", type=int, default=32)
    ap.add_argument("--no_msa", action="store_true",
                    help="THE CONTROL ARM. Run the SAME base checkpoint with NO MSA attached. "
                         "Without this, the 'with-MSA' run differs from ESMFold2-Fast in BOTH the "
                         "MSA and the checkpoint (base 1.3GB vs Fast 721MB are different models), "
                         "and nothing can be attributed to the MSA. This arm isolates it.")
    a = ap.parse_args()

    print(f"Loading {a.model} ...", flush=True)
    model = ESMFold2Model.from_pretrained(a.model).cuda().eval()
    model.set_chunk_size(a.chunk_size if a.chunk_size > 0 else None)
    cfg = getattr(model.config, "msa_encoder", None)
    enabled = getattr(cfg, "enabled", None) if cfg is not None else None
    print(f"  msa_encoder.enabled = {enabled}", flush=True)
    if not enabled and not a.no_msa:
        print("  *** THIS CHECKPOINT HAS NO MSA ENCODER — the with-MSA arm would be a no-op. ***")
        return
    print(f"  ARM: {'CONTROL (no MSA attached)' if a.no_msa else 'WITH MSA'}", flush=True)
    builder = ESMFold2InputBuilder()

    cxs = load_complexes(a.complexes, a.msa_max_depth, no_msa=a.no_msa)
    out_root = Path(a.out_dir)
    t0 = time.time()
    n = 0
    for seed in a.seeds:
        for name, sp in cxs:
            pred_dir = out_root / name / f"seed_{seed}" / "predictions"
            if (pred_dir / f"{name}_sample_{a.num_diffusion_samples-1}.cif").exists():
                continue                                   # resume-safe
            pred_dir.mkdir(parents=True, exist_ok=True)
            s = int(seed)
            torch.manual_seed(s); np.random.seed(s)
            try:
                res = builder.fold(
                    model, sp,
                    num_loops=a.num_loops,
                    num_sampling_steps=a.num_sampling_steps,
                    num_diffusion_samples=a.num_diffusion_samples,
                    lm_mask_pct=a.lm_mask_pct,
                    lm_dropout=a.lm_dropout,
                    msa_max_depth=a.msa_max_depth,
                )
            except Exception as e:
                print(f"  ! {name} seed {seed} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            # Write EXACTLY the layout run_foldbench_abag.py writes, so eval_ost_foldbench.py
            # scores this run with no changes. The coordinates live on `sample.complex`, not on
            # the result object itself.
            samples = res if isinstance(res, list) else [res]
            for k, r in enumerate(samples[:a.num_diffusion_samples]):
                (pred_dir / f"{name}_sample_{k}.cif").write_text(r.complex.to_mmcif())
                iptm = r.iptm if r.iptm is not None else (r.ptm or 0.0)
                conf = dict(ranking_score=iptm, iptm=r.iptm, ptm=r.ptm,
                            chain_pair_iptm=(r.pair_chains_iptm.tolist()
                                             if r.pair_chains_iptm is not None else None))
                (pred_dir / f"{name}_summary_confidence_sample_{k}.json").write_text(json.dumps(conf))
                if r.plddt is not None:
                    np.save(pred_dir / f"{name}_plddt_sample_{k}.npy", r.plddt.half().cpu().numpy())
                if r.pae is not None:
                    np.save(pred_dir / f"{name}_pae_sample_{k}.npy", r.pae.half().cpu().numpy())
            n += 1
            if n % 5 == 0:
                print(f"  [{n}] {name} seed {seed}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\ndone: {n} (complex, seed) folds in {time.time()-t0:.0f}s -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
