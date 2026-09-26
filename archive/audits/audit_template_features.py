"""Audit: does the model actually receive the donor pose as its template?

Runs ONLY Protenix's inference data pipeline (CPU, no model, no GPU) on one templated job, with the
same template source and environment folder.py sets, and dumps the template features the model would
receive (TMPL_DUMP_INFER). Then compares them against the donor cif read independently with gemmi:

    residue types   template_aatype vs the query sequence, per token
    coordinates     template CA (or C1'/first atom) vs the donor's CA -- exact, since TMPL_KEEP_FRAME=1
                    leaves the template in the donor's own frame
    coverage        fraction of query tokens the template covers

Usage (repo root):
    CUDA_VISIBLE_DEVICES= $PROTENIX_PYTHON archive/audits/audit_template_features.py \
        --ns experimental_data/self_template_fixed_50_seed/templates --job 8ror_AB_A00 --out <dir>
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
sys.path.insert(0, ROOT)
import paths  # noqa: E402

sys.path.insert(0, str(paths.PROTENIX_SRC))
os.environ.setdefault("LAYERNORM_TYPE", "openfold")


def dump_features(ns, job, out):
    """in: namespace, job name, output dir.  out: path of the npz of inference template features."""
    from collections.abc import Mapping

    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config import parse_configs
    from protenix.data.inference.infer_dataloader import InferenceDataset
    from runner.inference import model_configs

    sys.path.insert(0, os.path.join(ROOT, "diffusion_commit_clean_v2"))
    from folder import TMPL_ENV

    chosen = json.load(open(f"{ns}/jobs/{job}/chosen.json"))
    # the same environment folder.py applies to a templated unit
    os.environ.update(TMPL_ENV)
    os.environ["PROTENIX_MAX_TEMPLATES"] = str(len(chosen))
    os.environ["SELFTMPL_MMCIF_DIR"] = f"{ns}/mmcif"
    os.environ["SELFTMPL_RELEASE_DATES"] = f"{ns}/release_dates.json"
    npz = f"{out}/{job}_template_features.npz"
    os.environ["TMPL_DUMP_INFER"] = npz

    base = {**configs_base, **{"data": data_configs}, **inference_configs}

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base, model_configs["protenix-v2"])
    configs = parse_configs(configs=base, arg_str="", fill_required_with_null=True)
    configs.input_json_path = f"{ns}/jobs/{job}/job.json"
    configs.dump_dir = out
    configs.use_template = True
    configs.data.template.prot_template_mmcif_dir = f"{ns}/mmcif"
    configs.data.template.release_dates_path = f"{ns}/release_dates.json"
    ds = InferenceDataset(configs=configs)
    ds[0]
    return npz, chosen


def compare(npz, chosen, ns, job):
    """in: dumped features, chosen.json.  out: printed per-chain comparison against the donor cif."""
    import gemmi
    import numpy as np

    from protenix.data.constants import PROTEIN_TYPES_ONE_LETTER as AA

    f = dict(np.load(npz))
    job_json = json.load(open(f"{ns}/jobs/{job}/job.json"))[0]["sequences"]
    queries = [e["proteinChain"]["sequence"] for e in job_json if "proteinChain" in e]
    assert all("proteinChain" in e and e["proteinChain"].get("count", 1) == 1 for e in job_json), \
        "audit assumes one protein copy per entity"
    ok = True
    for t, dn in enumerate(chosen):
        st = gemmi.read_structure(dn["cif"])
        donor = {ch.name: ch for ch in st[0]}
        aat, pos, msk = f["template_aatype"][t], f["template_atom_positions"][t], f["template_atom_mask"][t]
        off = 0
        for pi, q in enumerate(queries):
            ch = donor[dn["chains"][pi]]      # template chain pinned to this entity by build_templates
            L = len(q)
            got = "".join(AA[a] if a < len(AA) else "?" for a in aat[off:off + L])
            dseq = gemmi.one_letter_code([r.name for r in ch])
            aa_match = sum(a == b for a, b in zip(got, q))
            # every template atom must sit exactly on an atom of the same donor residue
            worst, n_atoms, covered = 0.0, 0, 0
            for i, res in enumerate(ch):
                tp = pos[off + i][msk[off + i]]
                if len(tp) == 0:
                    continue
                covered += 1
                dp = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res])
                d = np.linalg.norm(tp[:, None, :] - dp[None, :, :], axis=-1).min(1)
                worst = max(worst, float(d.max()))
                n_atoms += len(tp)
            good = (aa_match == L and dseq == q and len(ch) == L and covered == L and worst < 1e-2)
            ok &= good
            print(f"template {t} chain {dn['chains'][pi]} (entity {pi}, {L} res): "
                  f"aatype==query {aa_match}/{L} | donor seq==query {dseq == q} | "
                  f"residues with coords {covered}/{L} | atoms {n_atoms} | "
                  f"worst atom offset from donor {worst:.4f} A | {'OK' if good else 'MISMATCH'}")
            off += L
        assert off == aat.shape[0], f"token count {aat.shape[0]} != sum of chains {off}"
    print("ALL OK" if ok else "MISMATCHES FOUND")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    ns = os.path.abspath(A.ns)
    os.makedirs(A.out, exist_ok=True)
    npz, chosen = dump_features(ns, A.job, os.path.abspath(A.out))
    compare(npz, chosen, ns, A.job)
