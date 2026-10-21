"""LoRA adapters for the confidence head.

WHY THIS EXISTS. FT3 trains the whole confidence head -- 19.8M parameters -- on 488 structures,
and the 30-structure anchor showed what that capacity does with too little data: training PAE
fell 38% while held-out PAE rose 51%. On the full set the gap is smaller but still opening
(epochs 1-6: train -7.6%, held-out -0.9%). LoRA constrains each weight's UPDATE to rank r
instead of full rank, which is a capacity limit on what finetuning can express rather than a
penalty applied after the fact.

  W_eff = W_frozen + (alpha / r) * B @ A        A: (r, in)   B: (out, r)

A is Kaiming-normal and B IS ZERO, so at step 0 the adapted model is BIT-IDENTICAL to the
shipped head -- the same discipline pathway C used for its zero-init direction tensor, and the
thing that makes "LoRA on" vs "LoRA off" a clean single variable at step 0.

WHAT GETS ADAPTED. Any submodule holding a 2-D parameter called `weight`, matched by name
against --lora-targets. Keying on the class name would miss most of the head: 54 modules are
Linear/LinearNoBias, but the ~96 matrices inside tri_mul_* and tri_att_* are a different class
and carry 1.57M parameters between them.

BIAS IS NOT ADAPTED and not unfrozen. Every matched module in this head is bias-free, so there
is nothing to decide; if one ever has a bias it stays frozen and that is reported, rather than
being silently trained and quietly making "LoRA" mean something else.
"""
import math
import re

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen linear with a rank-r additive update.

    DELEGATES RATHER THAN REIMPLEMENTS. The base module is kept and called, so whatever it does
    -- LinearNoBias, a custom class, a bias we are not touching -- keeps happening exactly as
    before and the adapter is strictly additive on top.
    """

    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        w = base.weight
        if w.dim() != 2:
            raise ValueError(f"LoRA needs a 2-D weight, got {tuple(w.shape)}")
        out_f, in_f = w.shape
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if rank > min(in_f, out_f):
            raise ValueError(f"rank {rank} exceeds min(in={in_f}, out={out_f}); the adapter would "
                             f"have more parameters than the weight it adapts")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, dtype=w.dtype, device=w.device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=w.dtype, device=w.device))
        # A IS RANDOM AND B IS ZERO. Both zero gives a dead gradient -- dL/dA goes through B and
        # dL/dB goes through A, so the product stays zero forever. Both random breaks the
        # bit-identical start. This is the only pairing that gives neither.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x, *args, **kwargs):
        y = self.base(x, *args, **kwargs)
        # IN THE BASE WEIGHT'S DTYPE, not the input's. The trunk runs in bf16 under autocast and
        # the adapter must land in the same dtype as what it is added to, or this raises on the
        # add instead of training.
        h = self.drop(x).to(self.lora_A.dtype)
        return y + (h @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

    @torch.no_grad()
    def merged_weight(self):
        """The base weight with the adapter folded in. Used to write a plain head checkpoint."""
        return self.base.weight + (self.lora_B @ self.lora_A) * self.scaling


def _targets_ok(name: str, patterns):
    return any(re.search(p, name) for p in patterns)


def iter_adaptable(root: nn.Module):
    """(name, module) for every submodule carrying a 2-D parameter named `weight`."""
    for name, m in root.named_modules():
        if isinstance(m, LoRALinear):
            continue
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2 and isinstance(w, nn.Parameter):
            yield name, m


def apply_lora(root: nn.Module, rank: int, alpha: float, targets, dropout: float = 0.0):
    """Replace matched submodules of `root` with LoRALinear. Returns (param_names, report).

    `targets` is a list of regexes matched against the module name RELATIVE TO root. An empty
    list matches everything adaptable.
    """
    patterns = [p for p in (targets or []) if p]
    chosen = [(n, m) for n, m in iter_adaptable(root)
              if (not patterns or _targets_ok(n, patterns))
              and rank <= min(m.weight.shape)]
    skipped_rank = [(n, tuple(m.weight.shape)) for n, m in iter_adaptable(root)
                    if (not patterns or _targets_ok(n, patterns))
                    and rank > min(m.weight.shape)]
    if not chosen:
        raise SystemExit(
            "STOP: --lora-rank is set but no module matched --lora-targets, so the optimiser "
            "would be empty and the run would train nothing while reporting success.\n"
            f"  patterns: {patterns or ['<all>']}\n"
            f"  adaptable modules: {sum(1 for _ in iter_adaptable(root))}")
    with_bias = []
    for name, m in chosen:
        parent = root
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        if getattr(m, "bias", None) is not None:
            with_bias.append(name)
        setattr(parent, parts[-1], LoRALinear(m, rank, alpha, dropout))
    # FREEZE EVERYTHING ELSE. Done after the swap so the base weights now inside LoRALinear stay
    # frozen too: the point of LoRA here is that the only trainable tensors are the adapters.
    for p in root.parameters():
        p.requires_grad_(False)
    names = []
    for n, p in root.named_parameters():
        if n.endswith(".lora_A") or n.endswith(".lora_B"):
            p.requires_grad_(True)
            names.append(n)
    base_n = sum(p.numel() for _n, p in root.named_parameters()
                 if not (_n.endswith(".lora_A") or _n.endswith(".lora_B")))
    lora_n = sum(p.numel() for _n, p in root.named_parameters() if _n in set(names))
    report = {"modules_adapted": len(chosen), "lora_tensors": len(names),
              "lora_params": lora_n, "frozen_params": base_n,
              "fraction_of_head": lora_n / max(1, base_n),
              "skipped_rank_too_high": skipped_rank, "adapted_with_bias": with_bias,
              "rank": rank, "alpha": alpha, "dropout": dropout,
              "targets": patterns or ["<all>"]}
    return names, report


@torch.no_grad()
def merge_and_unload(root: nn.Module):
    """Fold every adapter into its base weight and put the plain modules back.

    This is what makes a LoRA run scorable by code that knows nothing about LoRA: the checkpoint
    written afterwards has exactly the tensor names and shapes the shipped head has.
    """
    merged = 0
    for name, m in list(root.named_modules()):
        if not isinstance(m, LoRALinear):
            continue
        parent = root
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        m.base.weight.copy_(m.merged_weight())
        for p in m.base.parameters():
            p.requires_grad_(True)
        setattr(parent, parts[-1], m.base)
        merged += 1
    return merged


def lora_state_dict(root: nn.Module):
    return {n: p.detach().cpu() for n, p in root.named_parameters()
            if n.endswith(".lora_A") or n.endswith(".lora_B")}


@torch.no_grad()
def merged_head_tensors(root: nn.Module):
    """{original_param_name: tensor} with every adapter folded in, WITHOUT mutating the model.

    THIS IS WHAT KEEPS A LoRA RUN SCORABLE. Wrapping renames `blk.linear.weight` to
    `blk.linear.base.weight`, so a checkpoint written straight from named_parameters() would not
    load into a plain head and phase_k_eval -- which knows nothing about LoRA -- would refuse it
    or, worse, silently miss tensors. Here the `.base` segment is stripped and the adapted
    weights come out merged, so the file is indistinguishable from a full-finetune checkpoint.

    Not merge_and_unload(): that mutates, and training continues after a checkpoint is written.
    """
    lora_names = {n for n, m in root.named_modules() if isinstance(m, LoRALinear)}
    out = {}
    for name, m in root.named_modules():
        if isinstance(m, LoRALinear):
            out[f"{name}.weight"] = m.merged_weight().detach().cpu()
    for n, p in root.named_parameters():
        if n.endswith(".lora_A") or n.endswith(".lora_B"):
            continue
        if n.endswith(".base.weight") and n[: -len(".base.weight")] in lora_names:
            continue                      # already emitted, merged, under its original name
        # a bias or other tensor living inside an adapted module keeps its real name minus .base
        out[n.replace(".base.", ".")] = p.detach().cpu()
    return out
