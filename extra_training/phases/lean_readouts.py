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

    head.memory_efficient_forward = types.MethodType(_lean_memory_efficient_forward, head)
    # ASSERT IT ACTUALLY TOOK. A flag that prints a line and changes nothing is the failure mode
    # --blocks-per-ckpt had for its whole life (phase_k_train.py:1333).
    if getattr(head.memory_efficient_forward, "__func__", None) is not \
            _lean_memory_efficient_forward:
        raise SystemExit("STOP: --lean-readouts did not take: the head's "
                         "memory_efficient_forward is not the lean mirror after assignment.")
    return sha
