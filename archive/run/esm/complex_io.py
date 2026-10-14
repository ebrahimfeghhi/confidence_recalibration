"""Shared complex-loading + deterministic input-conditioning helpers for ESMFold2.

Extracted from the (now-archived) per-loop diffusion experiment so the steering
engine doesn't depend on an experiment script. Used by steer/core/forward_grad.py
and steer/experiments/decode_dockq.py.
"""
import json
import string
from pathlib import Path

import torch
import torch.nn.functional as F

from esm.models.esmfold2 import ProteinInput, StructurePredictionInput
from transformers.models.esmfold2.modeling_esmfold2 import (
    NUM_RES_TYPES, MAX_ATOMIC_NUMBER, CHAR_VOCAB_SIZE,
)

CHAIN_LETTERS = string.ascii_uppercase


def load_complex(json_dir, name):
    """Find the complex `name` in the FoldBench JSONs -> StructurePredictionInput."""
    for json_path in sorted(Path(json_dir).glob("*.json")):
        if "updated" in json_path.name or "update-msa" in json_path.name:
            continue
        data = json.load(open(json_path))
        entry = data[0]
        if entry["name"] != name:
            continue
        chain_idx, inputs = 0, []
        for seq_entry in entry["sequences"]:
            if "proteinChain" not in seq_entry:
                continue
            cd = seq_entry["proteinChain"]
            count = cd.get("count", 1)
            ids = [CHAIN_LETTERS[chain_idx + i] for i in range(count)]
            chain_idx += count
            inputs.append(ProteinInput(id=ids if count > 1 else ids[0], sequence=cd["sequence"]))
        return StructurePredictionInput(sequences=inputs)
    raise SystemExit(f"Complex {name!r} not found in {json_dir}")


def build_conditioning(model, feat):
    """Replicate forward()'s deterministic input prep (no LM, no trunk).

    Returns the kwargs bundle structure_head.sample / confidence_head need;
    identical across seeds and loops, so computed once."""
    tok_mask = feat["token_attention_mask"]
    atm_mask = feat["atom_attention_mask"]
    res_type = feat["res_type"]

    if res_type.dim() == 2:
        res_type_oh = F.one_hot(res_type.long(), num_classes=NUM_RES_TYPES).float()
        res_type_oh = res_type_oh * tok_mask.unsqueeze(-1).float()
    else:
        res_type_oh = res_type.float()

    msa = feat.get("msa")
    if msa is not None:
        msa_oh = F.one_hot(msa.long(), num_classes=NUM_RES_TYPES).float()
        mam = feat.get("msa_attention_mask")
        if mam is not None:
            msa_oh = msa_oh * mam.float().unsqueeze(-1)
            cnt = mam.float().sum(dim=1).clamp(min=1)
            profile = msa_oh.sum(dim=1) / cnt.unsqueeze(-1)
        else:
            profile = msa_oh.mean(dim=1)
    else:
        profile = res_type_oh

    deletion_mean = feat.get("deletion_mean")
    if deletion_mean is None:
        deletion_mean = torch.zeros(res_type.shape[0], res_type.shape[1], device=res_type.device)

    ref_element_oh = F.one_hot(feat["ref_element"].long(), num_classes=MAX_ATOMIC_NUMBER).float()
    ref_name_oh = F.one_hot(feat["ref_atom_name_chars"].long(), num_classes=CHAR_VOCAB_SIZE).float()
    atm_f = atm_mask.float()
    ref_element_oh = ref_element_oh * atm_f.unsqueeze(-1)
    ref_name_oh = ref_name_oh * atm_f.unsqueeze(-1).unsqueeze(-1)
    atom_to_token = feat["atom_to_token"] * atm_mask.long()

    use_amp = res_type.device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
        x_inputs = model.inputs_embedder(
            aatype=res_type_oh, profile=profile.float(), deletion_mean=deletion_mean.float(),
            ref_pos=feat["ref_pos"], atom_attention_mask=atm_mask,
            ref_space_uid=feat["ref_space_uid"], ref_charge=feat["ref_charge"],
            ref_element=ref_element_oh, ref_atom_name_chars=ref_name_oh,
            atom_to_token=atom_to_token,
        )
        rel_pos = model.rel_pos(
            residue_index=feat["residue_index"], asym_id=feat["asym_id"],
            sym_id=feat["sym_id"], entity_id=feat["entity_id"], token_index=feat["token_index"],
        )
        token_bonds_encoding = model.token_bonds(feat["token_bonds"].float())
    pair_mask = tok_mask[:, :, None].float() * tok_mask[:, None, :].float()

    return dict(x_inputs=x_inputs, rel_pos=rel_pos, token_bonds_encoding=token_bonds_encoding,
                ref_element_oh=ref_element_oh, ref_name_oh=ref_name_oh,
                atom_to_token=atom_to_token, pair_mask=pair_mask)
