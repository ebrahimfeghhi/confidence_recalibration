#!/usr/bin/env python
"""PHASE J, step 2a -- instantiate Protenix's TRAINING dataset over our prepared data.

WHY THIS AND NOT OUR OWN LOADER. The cache has to hold what the FT3 loss consumes: ground-truth
coordinates in the model's atom order, the coordinate and representative-atom masks, the frames the
PAE loss aligns on. All of that comes out of Protenix's own dataset (`dataset.py:894` returns
`features_dict, labels_dict, label_full_dict`). Building it ourselves would be a second
implementation of the thing the comparison depends on.

Running this file instantiates the dataset and prints one item's tensors -- a CPU check that the
prepared data, the MSA store and the template store all wire together before any GPU time is spent.

CROP IS OFF. Protenix's FT3 crops to 768 tokens before the trunk; we fold whole complexes at
inference and the validation cache this will be compared against is uncropped, so `train_crop_size`
is set past the largest JOB -- 3,144 tokens measured over train_prep/job_indices.csv, hence 3,400.
A crop is resampled per draw, so any structure that crossed the limit would be non-replayable. This
is a deliberate deviation from the stage we are re-running, recorded in docs/cached_training.md.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTENIX = "/home/ebrahim/Protenix"
sys.path.insert(0, PROTENIX)


def build_configs(crop_size, indices, bio_dir, mmcif_dir, msa_root, seq_index, templates_root):
    from configs.configs_base import configs as base_configs
    from configs.configs_data import data_configs
    from configs.configs_model_type import model_configs
    from protenix.config import parse_configs
    from protenix.config.extend_types import ListValue

    name = "sabdab_ft3"
    data = {**data_configs}
    # train_sets is parsed as a comma-separated list and must line up 1:1 with
    # train_sampler.train_sample_weights (dataset.py:1197).
    data["train_sets"] = ListValue([name])
    # ListValue cannot be empty -- extend_types.py:42 infers the dtype from value[0].
    data["test_sets"] = ""
    data["train_sampler"] = {**data.get("train_sampler", {}),
                             "train_sample_weights": ListValue([1.0])}
    data[name] = {
        "base_info": {
            "mmcif_dir": mmcif_dir,
            "bioassembly_dict_dir": bio_dir,
            "indices_fpath": indices,
            "pdb_list": "",
            "random_sample_if_failed": False,   # a failure must surface, not be replaced
            "max_n_token": -1,
            "use_reference_chains_only": False,
            # ions excluded from SAMPLING, as Protenix's own weightedPDB config does
            # (configs_data.py:158). An empty list breaks parse_configs, which infers a list's type
            # from value[0]; this is their value, not a workaround.
            "exclusion": {"mol_1_type": ListValue(["ions"]), "mol_2_type": ListValue(["ions"])},
        },
        "cropping_configs": {
            "method_weights": ListValue([0.2, 0.4, 0.4]),
            "crop_size": crop_size,
        },
        "sampler_configs": {"sampler_type": "weighted"},
    }
    # RNA MSA OFF. The eligibility filter keeps only polypeptide antigens, so no RNA chain can
    # reach the featurizer, and leaving it on makes the dataset demand an rna_msa index we have no
    # reason to hold (file_io.py:253 raises on the missing json before any item is built).
    # An EMPTY RNA index, not a missing one. msa_featurizer.py:417 loads every rna json before
    # enable_rna_msa is consulted, so the flag alone does not spare us the file. Our set has no RNA
    # chain by construction, so an empty index is the truthful thing to hand it.
    data["msa"] = {**data["msa"], "enable_rna_msa": False,
                   "rna_seq_or_filename_to_msadir_jsons": ListValue([f"{ROOT}/train_prep/rna_seq_index_empty.json"]),
                   "prot_seq_or_filename_to_msadir_jsons": ListValue([seq_index]),
                   "prot_msadir_raw_paths": ListValue([msa_root])}
    data["template"] = {**data["template"],
                        "prot_seq_or_filename_to_templatedir_jsons": ListValue([seq_index]),
                        "prot_template_raw_paths": ListValue([templates_root])}
    cfg = {**base_configs, "data": data, **model_configs["protenix-v2"]}
    cfg["train_crop_size"] = crop_size
    return parse_configs(cfg, fill_required_with_null=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", default=f"{ROOT}/train_prep/job_indices.csv")
    ap.add_argument("--bio-dir", default=f"{ROOT}/train_prep/job_bioassembly")
    # THE JOB CIFS, matching the bioassemblies below. Pointing this at whole entries would describe
    # a different system from the one the poses come from.
    ap.add_argument("--mmcif-dir", default=f"{ROOT}/train_prep/job_cifs")
    ap.add_argument("--msa-root", default=f"{ROOT}/msa_work/mmcif_msa")
    ap.add_argument("--seq-index", default=f"{ROOT}/msa_work/pdb_seqs/seq_to_pdb_index.json")
    # PAST THE LARGEST JOB, measured, not guessed. The job-level bioassemblies run to 3,144 tokens
    # (train_prep/job_indices.csv), so 3,072 would have cropped exactly one structure -- and a crop is
    # resampled per draw, which would make that unit non-replayable.
    ap.add_argument("--crop-size", type=int, default=3400, help="past the largest job = no crop")
    ap.add_argument("--index", type=int, default=0, help="which datapoint to materialise")
    a = ap.parse_args()
    cfg = build_configs(a.crop_size, a.indices, a.bio_dir, a.mmcif_dir, a.msa_root, a.seq_index,
                        a.msa_root)
    from protenix.data.pipeline.dataset import get_datasets
    err = os.path.join(ROOT, "train_prep", "dataset_errors")
    os.makedirs(err, exist_ok=True)
    train, _test = get_datasets(cfg, err)
    print(f"datapoints: {len(train)}")
    # dataset.py:590 returns a DICT -- input_feature_dict / label_dict / label_full_dict / basic --
    # not the (features, labels, label_full) tuple that process_one's inner helper returns at :894.
    item = train[a.index]
    print(f"item keys: {sorted(item)}")
    print(f"basic: { {k: v for k, v in item['basic'].items() if not hasattr(v, '__len__') or isinstance(v, str)} }")
    for nm in ("input_feature_dict", "label_dict", "label_full_dict"):
        d = item[nm]
        ks = sorted(d) if isinstance(d, dict) else []
        print(f"\n{nm}: {len(ks)} keys")
        for k in ks:
            v = d[k]
            print(f"   {k:34s} {tuple(getattr(v, 'shape', ()))} {getattr(v, 'dtype', type(v).__name__)}")


if __name__ == "__main__":
    main()
