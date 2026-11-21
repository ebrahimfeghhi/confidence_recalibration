"""Stop the confidence head from COMPUTING the three readouts --pae-only already throws away.

WHAT --pae-only ALREADY DOES, AND WHAT IT DOES NOT. phase_k_train.py replaces plddt_loss,
pde_loss and resolved_loss with stubs and skips the all-atom lddt, so none of those three
TERMS is trained. The HEAD is untouched: ConfidenceHead.memory_efficient_forward still builds
pde_pred, plddt_pred and resolved_pred on every pose, and phase_k_train.py puts all four into
`pred`. Nothing in the loss reads three of them -- phases/rank_loss.py:238 takes pae logits
alone -- so backward never traverses those branches, which is exactly why they are expensive:
a branch backward does not walk is not freed by backward either. Their saved activations live
until _one_pose returns, straight through the window where the peak is set.

WHAT IT COSTS, from the shapes in the live checkpoint (c_z 256, c_s 384, b_pde 64, b_plddt 50,
max_atoms_per_token 24), all fp32:

    z_pair + z_pair.transpose(-2, -3)      N_tok^2 * 1024 B
    pde_ln(...)                            N_tok^2 * 1024 B
    pde_pred                               N_tok^2 *  256 B
    plddt_weight[atom_to_tokatom_idx]      N_atom  * 76800 B     <-- the big one
    resolved_weight[...], a, 2 LayerNorms  N_atom  *  7680 B

which is ~1.3 GB at a median 617-token unit and ~6.5 GB at the 1543-token worst case, against
the 1.8 GB of headroom section 11 measured. THESE ARE ARITHMETIC, NOT MEASUREMENTS. The reason
to care about them is that they are large enough to be worth measuring, and
run_logging/bench_ckpt_by_size.py is what measures them.

WHY A MIRRORED COPY AND NOT A STUBBED SUBMODULE. Patching self.pde_ln would not help: the
expression is `self.pde_ln(z_pair + z_pair.transpose(-2, -3))`, so the N_tok^2 transpose-add is
allocated BEFORE pde_ln is reached. Avoiding it means intercepting the whole function, which
means mirroring the part that is kept. A mirror can go stale, so install() refuses to run
against a Protenix whose version of that function is not the one this was written from.

WHAT THE PLACEHOLDERS MUST STILL SATISFY. protenix/model/loss.py:1574 calls EVERY entry of
loss_fns unconditionally -- the weight is read but the lambda runs regardless -- and the
plddt lambda evaluates `pred_dict["plddt"][..., coord_mask, :]` in its own body, before the
stubbed module is ever reached. So plddt cannot be a scalar: it has to keep an N_atom axis that
a coordinate_mask of N_atom booleans can index. It is returned as zeros [N_atom, b_plddt],
which is N_atom * 200 B -- 2.3 MB at 11,500 atoms, against the 884 MB weight gather that
produced it. pde and resolved are passed to their stubs whole and never indexed, so they stay
at one element, shaped only so ConfidenceHead.forward's torch.stack(dim=-4)/(dim=-3) is legal.
"""
import hashlib
import inspect
import types

import torch

from protenix.model.utils import one_hot

# THE EXACT memory_efficient_forward THIS MIRRORS. Recorded against
# /home/ebrahim/Protenix/protenix/model/modules/confidence.py on 2026-11-01. If Protenix's
# version changes, the mirror below is stale by definition: it could keep computing the old pae
# while the rest of the stack moved, and the arm would be compared against a head that no
# longer exists. That is a hard stop and not a warning, because the failure is silent.
PINNED_SHA = "22df48da4e0ebd20f101f43dc0942fa0e13d91ba3d98b4bc5b963b77f2a0dc04"
PINNED_WHERE = "ConfidenceHead.memory_efficient_forward"

# ---- THE SECOND HALF OF THE SAME JOB: THE COMPUTE THAT FED THE DEAD READOUTS -----------------
#
# WHAT WAS STILL RUNNING. The mirror above stops COMPUTING pLDDT, PDE and resolved. It does not stop
# computing the thing two of them read. PairformerBlock.forward updates `z` through tri_mul_out,
# tri_mul_in, tri_att_start, tri_att_end and pair_transition -- every one a function of `z` ALONE --
# and only then updates `s` through attention_pair_bias and single_transition, both of which read
# `z`. So THE DEPENDENCY IS STRICTLY ONE-DIRECTIONAL: z never depends on s. Under --pae-only the
# readout is linear_no_bias_pae(pae_ln(z_pair)), which reads z only, and `s` feeds nothing but the
# pLDDT and resolved heads the mirror already stubs. The mirror's own comment says as much -- "s_single
# is now unused by anything that is kept" -- and then keeps computing it in all four blocks, for
# every pose of every unit of every epoch.
#
# THE SWITCH ALREADY EXISTS AND IS AN INSTANCE ATTRIBUTE. PairformerBlock gates the single track on
# `if self.c_s > 0:` in BOTH __init__ (so the modules are not built) and forward (so they are not
# called). Setting `block.c_s = 0` on a built block therefore turns the pathway off without touching
# the module tree: the parameters stay where the checkpoint expects them, nothing is renamed, nothing
# is dropped from the state dict, and phase_k_eval's set-equality check on head tensors still passes.
# That is why this is done by attribute and NOT by rebuilding the stack with c_s=0.
#
# PairformerStack.forward DOES NOT TOUCH s EITHER -- it hands (s, z) to checkpoint_blocks and returns
# what comes back -- so with c_s=0 on every block, `s_single` is the input `s_trunk` unchanged. The
# mirror's `s_single.to(torch.float32)` therefore still works on a real tensor and still reaches
# nothing.
PINNED_BLOCK_SHA = "379f595070ca6467f33506b33869a7a7a5773893d362faa7a7530187ae950d78"
PINNED_BLOCK_WHERE = "PairformerBlock.forward"
PINNED_STACK_SHA = "5733f4f4930b37ba33147739f5856778948383ed576243257b155457e5df1076"
PINNED_STACK_WHERE = "PairformerStack.forward"

# THE PARAMETERS NOTHING READS UNDER --pae-only --lean-readouts, as name prefixes RELATIVE to the
# confidence head. Measured by instantiating the live configuration (n_blocks 4, c_z 256, c_s 384,
# c_s_inputs 449, b_pae 64, b_plddt 50, b_resolved 2, max_atoms_per_token 24, hidden_scale_up True,
# which reproduces the head's 19,752,270 parameters and a block's 4,749,184 exactly):
#
#     pairformer single pathway   10,053,120   50.90%   attention_pair_bias + single_transition x4
#     plddt_weight                   460,800    2.33%
#     resolved_weight                 18,432    0.09%
#     linear_no_bias_pde              16,384    0.08%
#     input_strunk_ln / plddt_ln /
#       resolved_ln / pde_ln            2,816    0.01%
#     ------------------------------------------------
#     dead                        10,551,552   53.42%   (live: 9,200,718)
#
# SO MORE THAN HALF THE "TRAINABLE" SET RECEIVES NO GRADIENT, and the project already knew: the
# --grad-probe path prints "of 258 tensors marked trainable, PAE reaches 100, the rel term reaches
# 100, and NEITHER reaches 158" and adds "EFFECTIVE trainable set is ... not ...". It diagnosed it and
# never acted on it.
#
# input_strunk_ln IS ONLY DEAD ONCE THE SINGLE PATHWAY IS, which is why it is in this list and not in
# the readout list: ConfidenceHead.forward applies it to s_trunk before the mirror is reached, so
# with the single track live it is the first op of a live path.
DEAD_UNDER_PAE_ONLY = (
    "plddt_weight", "resolved_weight", "plddt_ln.", "resolved_ln.",
    "linear_no_bias_pde.", "pde_ln.", "input_strunk_ln.",
)
DEAD_SINGLE_SUFFIXES = ("attention_pair_bias.", "single_transition.")


def _pin(fn, want, where):
    sha = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    if sha != want:
        raise SystemExit(
            f"STOP: --lean-single is pinned to a specific {where}.\n"
            f"  pinned    {want}\n"
            f"  installed {sha}\n"
            f"  at        {inspect.getsourcefile(fn)}\n"
            f"The guard this relies on is `if self.c_s > 0:` and the property it relies on is that "
            f"z never depends on s. Protenix's version of that function changed, so NEITHER is "
            f"established any more. Re-read it, confirm both, update the pin, and re-run "
            f"run_logging/test_lean_single.py before using this flag again.")
    return sha


def dead_head_params(head, single_dropped):
    """Head parameter names (relative to the head) that receive no gradient under --pae-only.

    `single_dropped` says whether the single pathway has been turned off; without that, the
    pairformer's attention_pair_bias and single_transition still RUN, and although nothing reads
    their output either, leaving them in the optimiser is the pre-existing behaviour and this
    function does not quietly change it.
    """
    pats = list(DEAD_UNDER_PAE_ONLY)
    out = []
    for n, _p in head.named_parameters():
        if any(n == p.rstrip(".") or n.startswith(p) for p in pats):
            out.append(n)
        elif single_dropped and n.startswith("pairformer_stack.blocks.") and \
                any(s in n for s in DEAD_SINGLE_SUFFIXES):
            out.append(n)
    return out


def drop_single_pathway(model):
    """Turn off the dead single track in every confidence-head pairformer block.

    Returns (n_blocks_changed, n_params_now_dead). Raises rather than warning on anything it
    cannot establish: a flag that prints a line and changes nothing is the failure mode
    --blocks-per-ckpt had for its whole life.
    """
    head = getattr(model, "confidence_head", None)
    if head is None:
        raise SystemExit("STOP: --lean-single: the model has no confidence_head.")
    stack = getattr(head, "pairformer_stack", None)
    blocks = list(getattr(stack, "blocks", []) or [])
    if not blocks:
        raise SystemExit("STOP: --lean-single: the head's pairformer_stack has no blocks, so there "
                         "is no single pathway to turn off and the flag would be a no-op.")
    _pin(type(blocks[0]).forward, PINNED_BLOCK_SHA, PINNED_BLOCK_WHERE)
    _pin(type(stack).forward, PINNED_STACK_SHA, PINNED_STACK_WHERE)
    changed = 0
    for b in blocks:
        if getattr(b, "c_s", 0) > 0:
            b.c_s = 0
            changed += 1
    # ASSERT IT TOOK, on every block, by reading the attribute the forward actually branches on.
    still = [i for i, b in enumerate(blocks) if getattr(b, "c_s", 0) > 0]
    if still:
        raise SystemExit(f"STOP: --lean-single did not take on block(s) {still}: c_s is still "
                         f"positive, so attention_pair_bias and single_transition would still run.")
    if not changed:
        raise SystemExit("STOP: --lean-single: every block already had c_s <= 0, so this flag "
                         "changed nothing. Something else has already patched the stack and the "
                         "two changes must be reconciled rather than stacked blindly.")
    dead = dead_head_params(head, single_dropped=True)
    n_dead = sum(p.numel() for n, p in head.named_parameters() if n in set(dead))
    return changed, n_dead


def _lean_memory_efficient_forward(
    self,
    input_feature_dict,
    s_trunk,
    z_pair,
    pair_mask,
    x_pred_rep_coords,
    triangle_multiplicative="torch",
    triangle_attention="torch",
    inplace_safe=False,
    chunk_size=None,
):
    """memory_efficient_forward with the pae path only. Every kept line mirrors the original.

    Returns (plddt_pred, pae_pred, pde_pred, resolved_pred) so the caller is unchanged; three of
    the four are placeholders.
    """
    if inplace_safe:
        # THE ORIGINAL HAS TWO BRANCHES HERE and the arm takes the out-of-place one
        # (phase_k_train.py passes inplace_safe=False). Mirroring only the branch the arm uses
        # is deliberate; running the other one through this copy would be untested code.
        raise SystemExit(
            "STOP: lean_readouts mirrors the inplace_safe=False branch of "
            f"{PINNED_WHERE} only, because that is the one the trainer takes. It was called "
            "with inplace_safe=True, so it would be running a path nothing has checked.")

    # --- kept, identical to the original ------------------------------------------------
    with torch.amp.autocast("cuda", enabled=False):
        x_pred_rep_coords = x_pred_rep_coords.to(torch.float32)
        distance_pred = torch.cdist(x_pred_rep_coords, x_pred_rep_coords)

    z_pair = z_pair + self.linear_no_bias_d(
        one_hot(x=distance_pred, lower_bins=self.lower_bins, upper_bins=self.upper_bins)
    )
    z_pair = z_pair + self.linear_no_bias_d_wo_onehot(distance_pred.unsqueeze(dim=-1))

    s_single, z_pair = self.pairformer_stack(
        s_trunk,
        z_pair,
        pair_mask,
        triangle_multiplicative=triangle_multiplicative,
        triangle_attention=triangle_attention,
        inplace_safe=inplace_safe,
        chunk_size=chunk_size,
    )

    # Upcast after pairformer
    z_pair = z_pair.to(torch.float32)
    s_single = s_single.to(torch.float32)

    with torch.amp.autocast("cuda", enabled=False):
        pae_pred = self.linear_no_bias_pae(self.pae_ln(z_pair))
    # --- end of the kept path -----------------------------------------------------------

    # DROPPED: pde_ln(z_pair + z_pair.transpose(-2, -3)) and linear_no_bias_pde over it;
    #          broadcast_token_to_atom + plddt_ln + plddt_weight gather + einsum;
    #          resolved_ln + resolved_weight gather + einsum.
    #
    # s_single is now unused by anything that is kept. It is NOT deleted: the pairformer
    # returns it, so the tensor exists either way, and dropping the reference here would not
    # free the stack's activations that produced it.
    atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
    n_atom = int(torch.as_tensor(atom_to_token_idx).reshape(-1).shape[0])
    kw = {"dtype": torch.float32, "device": z_pair.device}
    # plddt KEEPS ITS N_atom AXIS -- loss.py:1772 indexes it with coordinate_mask before the
    # stubbed module is reached. See the module docstring.
    plddt_pred = torch.zeros((n_atom, self.b_plddt), **kw)
    # pde and resolved are handed to their stubs whole. Shaped [1, 1, 1] / [1, 1] only so that
    # ConfidenceHead.forward's torch.stack(..., dim=-4) and (..., dim=-3) stay in range.
    pde_pred = torch.zeros((1, 1, 1), **kw)
    resolved_pred = torch.zeros((1, 1), **kw)
    return plddt_pred, pae_pred, pde_pred, resolved_pred


def install(model, require_pae_only=True, pae_only=False):
    """Replace the head's memory_efficient_forward with the pae-only mirror.

    Returns the sha256 it verified, so the caller can print what it pinned against.
    """
    if require_pae_only and not pae_only:
        raise SystemExit(
            "STOP: --lean-readouts without --pae-only. The three readouts it stops computing "
            "are the ones --pae-only stubs the loss terms for; without it, pLDDT, PDE and "
            "resolved ARE trained, and replacing them with zeros would silently train three "
            "terms against constant logits.")
    head = getattr(model, "confidence_head", None)
    if head is None:
        raise SystemExit("STOP: --lean-readouts: the model has no confidence_head to patch.")

    orig = getattr(type(head), "memory_efficient_forward", None)
    if orig is None:
        raise SystemExit(
            f"STOP: --lean-readouts: {type(head).__name__} has no memory_efficient_forward. "
            f"The function this mirrors does not exist in the installed Protenix.")
    sha = hashlib.sha256(inspect.getsource(orig).encode()).hexdigest()
    if sha != PINNED_SHA:
        raise SystemExit(
            f"STOP: --lean-readouts is pinned to a specific {PINNED_WHERE}.\n"
            f"  pinned   {PINNED_SHA}\n"
            f"  installed{'':1} {sha}\n"
            f"  at       {inspect.getsourcefile(orig)}\n"
            f"Protenix's version of that function changed, so phases/lean_readouts.py is a copy "
            f"of code that no longer runs. Re-read it, re-derive which lines feed pae, update "
            f"the mirror AND PINNED_SHA together, and re-run "
            f"run_logging/test_lean_readouts.py before using this flag again.")

    # THE DIMS THE PLACEHOLDERS ARE BUILT FROM MUST EXIST, checked here rather than at pose 0 of
    # unit 0 forty minutes in.
    for attr in ("b_plddt", "pae_ln", "linear_no_bias_pae", "pairformer_stack",
                 "linear_no_bias_d", "linear_no_bias_d_wo_onehot", "lower_bins", "upper_bins"):
        if not hasattr(head, attr):
            raise SystemExit(f"STOP: --lean-readouts: the head has no {attr}, which the mirror "
                             f"needs. The pin matched but the module tree did not.")

    # IT MUST BE THE BASE OF THE CHAIN, NOT THE TOP OF IT.
    #
    # pose_direction.py:234 and pose_perblock.py:232 each capture
    # `orig_mef = head.memory_efficient_forward` and replace the attribute with a wrapper that
    # calls what it captured, so the three patches form a chain whose order is its meaning.
    # Installing this one last overwrites the outermost wrapper and drops their injections. That
    # does not fail quietly -- the mirror still calls self.pairformer_stack, whose _prep_blocks is
    # also wrapped and raises _NoPose with the one-hot cache unset -- but "dies at pose 0" is not
    # a check, it is a symptom, and it only fires when --pose-perblock happens to be on. This
    # refuses the ordering outright. Found in review.
    for _marker, _mod in (("_pose_direction_installed", "--frame-direction / pose_direction"),
                          ("_pose_perblock_installed", "--pose-perblock / pose_perblock"),
                          # SAME CONTRACT, THIRD PATCH. pose_inblock wraps
                          # memory_efficient_forward and also replaces the stack's _prep_blocks, so
                          # installing the lean mirror after it would drop the wrapper AND leave
                          # _prep_blocks raising _NoPose with the geometry unset.
                          ("_pose_inblock_installed", "--pose-inblock / pose_inblock")):
        if getattr(head, _marker, False):
            raise SystemExit(
                f"STOP: --lean-readouts must be installed BEFORE {_mod}, which has already "
                f"wrapped memory_efficient_forward ({_marker} is set). Installing the lean "
                f"mirror now would replace that wrapper and silently drop its injection from "
                f"every pose. Move the lean_readouts.install() call above it in "
                f"phase_k_train.py.")

    head.memory_efficient_forward = types.MethodType(_lean_memory_efficient_forward, head)
    # ASSERT IT ACTUALLY TOOK. A flag that prints a line and changes nothing is the failure mode
    # --blocks-per-ckpt had for its whole life (phase_k_train.py:1333).
    if getattr(head.memory_efficient_forward, "__func__", None) is not \
            _lean_memory_efficient_forward:
        raise SystemExit("STOP: --lean-readouts did not take: the head's "
                         "memory_efficient_forward is not the lean mirror after assignment.")
    return sha
