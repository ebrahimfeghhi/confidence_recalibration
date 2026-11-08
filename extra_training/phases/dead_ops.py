"""DEAD OPERATION ELIMINATION: do not run a confidence-head module whose output is EXACTLY ZERO.

WHAT IS DEAD, AND WHY IT IS DEAD. The shipped checkpoint (/home/ebrahim/checkpoint/protenix-v2.pt,
whose head this project saves as head_init.pt) carries several pairformer modules whose EVERY tensor
-- layer norms, the q/k/v/o projections, the gates, the output projection -- sits in a band
4.904160e-37 .. 4.907298e-37. Measured on the shipped head: one such tensor takes 2 distinct values
in its high 16 bits (0x0326 and 0x8326, i.e. a FIXED exponent and a sign) and 13 distinct values in
its low 16. That is not a trained weight and it is not a denormal either -- fp32's smallest normal
is 1.1754944e-38, so these are ordinary normal floats. It is a constant bit pattern, and the pattern
is what makes the module inert:

    a = linear_a(layer_norm(z))     ~ 1e-37 * O(1)   -- representable
    b = linear_b(layer_norm(z))     ~ 1e-37 * O(1)   -- representable
    a * b                           ~ 1e-74          -- UNDERFLOWS fp32 (min subnormal 1.4e-45)
    linear_z(... )                  ~ 1e-111         -- 0, exactly

So the zero is produced by UNDERFLOW OF A PRODUCT of two floor-valued factors, not by a zero-init
output projection. (A FRESH head is zero for that other reason -- see THE TRAP below -- which is
exactly why nothing here trusts a weight threshold on its own.) The margin is enormous: the product
only stops underflowing once |z| exceeds ~7.6e13, so this is not a knife edge.

MEASURED ON THE SHIPPED HEAD, at 16 tokens, fp32 and bf16 autocast, max|out|:

    block 0   tri_mul_out 0.0   tri_mul_in 0.0   tri_att_start 0.0   tri_att_end 0.0
    block 1   tri_mul_out 53.1  tri_mul_in 54.8  tri_att_start 0.0   tri_att_end 0.0
    block 2   tri_mul_out 40.2  tri_mul_in  5.4  tri_att_start 0.0   tri_att_end 0.0
    block 3   tri_mul_out 128.9 tri_mul_in  0.4  tri_att_start 2.5   tri_att_end 0.0
    pair_transition  block 0 1.28   block 1 0.0   block 2 17.7   block 3 22.0

Nine of the sixteen O(N^3) pair operations are dead, and block 1's pair_transition (which is
O(N^2), not cubic) as well.

WHY SKIPPING THEM IS BIT-IDENTICAL AND NOT AN APPROXIMATION. PairformerBlock.forward consumes each
of these as `dropout_add_rowwise(z, module(z), p_drop, training)` -- which is literally `z + x` when
p_drop is 0.0 or the module is in eval (fused_ops.py:213) -- or, for the transition, as
`z = z + pair_transition(z)`. Adding an exact zero is the identity on every bit of z. Handing the
same `dropout_add_rowwise` a zeros tensor instead of the module's own zeros tensor is therefore the
same arithmetic on the same values, including the dropout draw: the stub's zeros go through the same
call, so the RNG stream is untouched even at a non-zero --pair-dropout. (The scale 1/(1-p) applied
to a zero is still zero.)

A STUB IN PLACE OF THE MODULE, NOT A THIRD MIRROR OF PairformerBlock.forward. There is already one
mirror of that function -- phases/pose_inblock.py, which pins its sha256 precisely because a mirror
goes stale without saying so -- and a second copy would be a second thing to drift. Instead each
dead module OBJECT is replaced in the tree by a husk whose forward returns `torch.zeros_like(z)`
immediately. The block's own forward is not touched at all, so this works identically for blocks
0-2 (which take the native forward) and for the injected block (which takes pose_inblock's mirror),
and it will keep working if Protenix reorders the block.

THE HUSK CARRIES THE ORIGINAL'S PARAMETERS, which is not cosmetic. `phase_k_eval` guards the head
checkpoint by SET EQUALITY over the head's tensor names, `--train-only` and LoRA select by name,
and prefix_cache derives its frozen-prefix set from `head.named_parameters()`. A stub with no
parameters would drop ~100 names out of the tree and every one of those would break or silently
narrow. So `_DeadStub` shares the original's `_parameters`, `_buffers` and `_modules` BY REFERENCE:
named_parameters(), named_modules(), state_dict() and `.to()` all behave exactly as before, on the
same tensor objects. install() asserts that, by comparing the head's state-dict key set before and
after.

WHAT IT DOES *NOT* SHARE is the hook dictionaries, deliberately: a forward hook registered on the
original must NOT fire from the husk, because "the stub is not executed" is the claim
run_logging/test_dead_ops.py has to be able to measure. For the same reason install() REFUSES to
stub a module that already has a hook on it -- stubbing would stop that hook firing, which is a
silent change to whatever registered it.

INSTALL ORDER -- READ THIS BEFORE MOVING THE CALL.

  1. lean_readouts.install() and drop_single_pathway()   FIRST, always. It REPLACES
     memory_efficient_forward rather than wrapping it, so anything that wraps must come after.
  2. pose_inblock.install()                              needs (1); enforced there by a marker.
  3. --train-only / LoRA / --lean-single set requires_grad.
  4. dead_ops.install()                                  MUST BE AFTER (3). Its central refusal is
     "a module I would stub is trainable", and that is read from requires_grad on the tensors --
     which is not final until (3) has run. Installed any earlier it would be checking a flag's
     intent instead of the model's state, which is the whole distinction the refusal exists to make.
  5. prefix_cache.install()                              AFTER (4), though either order is correct.
     The husk leaves named_parameters() and named_modules() unchanged, so the cache's prefix set and
     its dropout scan see exactly what they saw before, and no parameter VALUE moves, so its digest
     is the same either way. dead_ops goes first only so that a dead_ops refusal happens before the
     cache turns on, and so the cache's log line describes the tree that will actually run.

HOW IT COMPOSES WITH THE PREFIX CACHE, stated because the honest number matters. On a cache HIT
blocks 0..block_index-1 are replaced by identities and the cut block's two tri-muls are skipped, so
on that pass the stubs for those modules were never going to run anyway; the only stub that still
earns anything on a hit is the cut block's tri_att_end. The two are therefore COMPLEMENTARY rather
than redundant: the cache removes the frozen prefix from pass B, and dead_ops removes half of pass
A -- the fill pass, which the cache leaves running in full.

A NOTE ON WHAT THIS IS NOT. It is not a claim that the shipped head is half untrained; it is a
claim that half of its pair operations contribute exactly nothing to the number this project
fine-tunes, and that paying for them is optional. Anything the arm TRAINS is excluded by (4) above.
"""
import threading

import torch

# THE PARAMETER FLOOR. Every tensor of a candidate must be below this for the candidate to be
# considered at all. It is a long way above the measured 4.9e-37 and a long way below anything a
# trained weight reaches (the smallest live max|w| on the shipped head is 0.59, block 3's
# tri_mul_in), so its exact value is not load-bearing. IT IS NOT SUFFICIENT ON ITS OWN -- see
# THE TRAP in run_logging/test_dead_ops.py -- which is why every candidate is also RUN.
FLOOR = 1e-30

# THE CANDIDATE SUBMODULES OF A PairformerBlock, and how each is called. The kwargs mirror
# PairformerBlock.forward's inplace_safe=False branch (pairformer.py:173-218), which is the branch
# the trainer takes, so the probe exercises the call the block will actually make.
#
# HARDCODED NAMES ARE CONSERVATIVE HERE, NOT WRONG: a sixth pair operation added upstream would be
# left running rather than wrongly skipped. The CALL SIGNATURE is the part that can drift, which is
# why install() pins PairformerBlock.forward's sha256 (reusing pose_inblock's pin) -- a renamed
# kwarg would otherwise silently change what the probe probes.
CANDIDATES = ("tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end", "pair_transition")

# WHICH KERNEL THE PROBE MUST RUN -- AND WHY IT IS AN ARGUMENT AND NOT THE STRING "torch".
#
# This was a real bug, caught in review. The first version of this file hardcoded
# triangle_multiplicative="torch" / triangle_attention="torch" and claimed to be "the call the
# block will actually make". IT IS NOT. phase_k_train.py passes configs.triangle_multiplicative and
# configs.triangle_attention, and configs_base.py:129-130 sets BOTH to "cuequivariance". With this
# arm's hidden_scale_up=True, c_hidden_mul == c_z == 256, so triangular.py:494's gate
# (`cuequivariance and c_z == c_hidden`) is OPEN and every tri-mul on a GPU takes the fused Triton
# kernel. Triangle attention likewise dispatches to cuequivariance_triangular_attn at layers.py:450,
# with a DIFFERENT qkv scaling (apply_scale is False for cueq, layers.py:434). So the probe was
# certifying the fp32 reference implementation while the thing being replaced was a fused kernel
# that had never been run -- exactly the "hardcoded verdict" this module's whole design rejects.
#
# The kernel names therefore come from the caller, who has `configs`. An unrecognised name is a
# hard stop rather than a silent fallback to "torch".
KERNELS_MUL = ("torch", "cuequivariance")
KERNELS_ATT = ("torch", "triattention", "cuequivariance", "deepspeed")


def _probe_call(sub, mod, z, mul, att):
    if sub in ("tri_mul_out", "tri_mul_in"):
        return mod(z, mask=None, inplace_safe=False, _add_with_inplace=False,
                   triangle_multiplicative=mul)
    if sub in ("tri_att_start", "tri_att_end"):
        return mod(z, mask=None, triangle_attention=att, inplace_safe=False, chunk_size=None)
    return mod(z)


# TWO INPUT SCALES. A module could in principle be zero at |z| ~ 1 and not at |z| ~ 1e3. BOTH
# tri-mul and tri-att open with a LayerNorm on z (triangular.py:525, :679), so |z| is normalised
# away before any weight is touched and the real margin lives in the weight magnitude, not in z --
# i.e. the x1e3 sweep buys less than it looks like it does. It is kept because pair_transition and
# any future candidate need not begin with a norm, and because a cheap probe that cannot mislead is
# worth more than the argument that it is unnecessary.
_PROBE_SCALES = (1.0, 1.0e3)

# ABOVE 16, AND THAT IS LOAD-BEARING. layers.py:438 is `if q.shape[-2] <= 16: triangle_attention =
# "torch"` -- so at the 8 tokens this used to probe, the cuequivariance attention kernel is
# UNREACHABLE and passing its name would still have measured the torch path. 24 is the smallest
# round number that enters the real dispatch.
_PROBE_TOKENS = 24

_LOCK = threading.Lock()
_S = {
    "on": False,
    "head": None,
    "orig": {},          # (block_index, sub) -> the module object taken out of the tree
    "stubs": {},         # (block_index, sub) -> the husk put in its place
    "params": [],        # every parameter of every stubbed module, for the step-boundary re-check
    "names": [],         # their names under the head, for the message
    "digest": None,      # (sum, max) over those parameters at install
    "calls": 0,          # how many times a stub answered -- "did the flag actually fire?"
    "report": [],
}


class _DeadStub(torch.nn.Module):
    """A husk of a module whose output is exactly zero: same parameters, no computation.

    THE PARAMETER / BUFFER / CHILD DICTS ARE SHARED BY REFERENCE with the module this replaces, so
    the head's named_parameters(), named_modules() and state_dict() are unchanged and `.to()`,
    `.float()` and `requires_grad_()` still reach the same tensor objects through the husk. The HOOK
    dicts are NOT shared -- see the module docstring.
    """

    def __init__(self, orig, where):
        super().__init__()
        self._dead_where = str(where)
        self._dead_orig_type = type(orig).__name__
        for attr in ("_parameters", "_buffers", "_modules", "_non_persistent_buffers_set",
                     "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks",
                     "_load_state_dict_post_hooks"):
            if hasattr(orig, attr):
                object.__setattr__(self, attr, getattr(orig, attr))
        self.training = bool(orig.training)

    def forward(self, x=None, *args, **kw):
        # THE IN-PLACE BRANCHES ARE A DIFFERENT CONTRACT AND ARE REFUSED. With
        # _add_with_inplace=True a TriangleMultiplicativeUpdate returns z + update, NOT the update
        # (triangular.py, and PairformerBlock.forward's inplace_safe branch assigns `z = ...`
        # rather than adding). A zeros return there would WIPE z instead of leaving it alone --
        # silently, with the right shape and dtype. lean_readouts and pose_inblock both already
        # refuse inplace_safe=True; this is the third lock on the same door, at the one place that
        # would be wrong rather than merely unchecked.
        if kw.get("inplace_safe") or kw.get("_add_with_inplace"):
            raise SystemExit(
                f"STOP: --skip-dead-ops stubbed {self._dead_where} and it was called with "
                f"inplace_safe={kw.get('inplace_safe')!r} / "
                f"_add_with_inplace={kw.get('_add_with_inplace')!r}. In that branch the module "
                f"returns the UPDATED z, not the update, so returning zeros would erase the pair "
                f"representation rather than leave it unchanged. The trainer passes "
                f"inplace_safe=False; whatever reached this call is a path nothing has checked.")
        if x is None:
            x = kw.get("z", kw.get("x"))
        if not torch.is_tensor(x):
            raise SystemExit(
                f"STOP: --skip-dead-ops stubbed {self._dead_where} and it was called without a "
                f"tensor as its first argument ({type(x).__name__}). The stub returns "
                f"zeros_like(that tensor), so it cannot answer a call shaped differently from "
                f"{self._dead_orig_type}.forward.")
        with _LOCK:
            _S["calls"] += 1
        # A BROADCAST ZERO, NOT zeros_like, AND THAT IS A MEASURED CHOICE. The consumer is
        # dropout_add_rowwise / `z + x`, which broadcasts, so both are the same arithmetic -- but a
        # materialised zeros_like is an [N_token, N_token, c_z] allocation, and at 256 tokens it
        # measured +64.0 MiB of tensors held across the backward (282.94 -> 346.94 MiB of
        # saved-for-backward bytes, a 22.6% REGRESSION) for the one stubbed module that is inside
        # the autograd graph. A zero-strided expand of a scalar is O(1) storage and that regression
        # disappears. The dtype and device come from the input, which is what makes the add
        # bit-identical; the SHAPE is carried by the broadcast rather than by the storage.
        return torch.zeros((), dtype=x.dtype, device=x.device).expand_as(x)

    def extra_repr(self):
        return f"DEAD (stubbed by dead_ops; was {self._dead_orig_type})"


def _probe_zero(mod, sub, c_z, autocast_dtype=None, triangle_multiplicative="torch",
                triangle_attention="torch"):
    # THE SAME KWARG NAMES scan()/install() TAKE, so a caller (including the test) cannot
    # reach this with the names transposed or silently fall back to "torch".
    mul, att = triangle_multiplicative, triangle_attention
    """Run the module and report whether its output is EXACTLY zero, in every regime tried.

    Returns (all_zero, detail) where detail is a list of (regime, max|out|, shape_ok, dtype_ok).

    SHAPE AND DTYPE ARE PART OF THE QUESTION, not an afterthought. The stub returns
    zeros_like(input); if the real module returned a different dtype the add it feeds would promote
    and the result would NOT be bit-identical, so a module whose output dtype does not track its
    input's is not a candidate however zero it is.
    """
    ref = next((p for p in mod.parameters()), None)
    if ref is None:
        return False, [("no-parameters", float("nan"), False, False)]
    dev, dt = ref.device, ref.dtype
    g = torch.Generator(device="cpu").manual_seed(20261107)
    base = torch.randn(_PROBE_TOKENS, _PROBE_TOKENS, int(c_z), generator=g).to(device=dev, dtype=dt)
    was_training = bool(mod.training)
    detail, all_zero = [], True
    try:
        # BOTH MODES. Transition (pair_transition) branches on self.training and takes a different
        # code path in eval, so probing only the mode the module happens to be in would leave the
        # other one unmeasured -- and the trainer and the evaluator are in different modes.
        for mode in (True, False):
            mod.train(mode)
            for scale in _PROBE_SCALES:
                regimes = [(f"{'train' if mode else 'eval'}/x{scale:g}", None)]
                if autocast_dtype is not None:
                    regimes.append((f"{'train' if mode else 'eval'}/x{scale:g}/"
                                    f"{str(autocast_dtype).replace('torch.', '')}",
                                    autocast_dtype))
                for label, ac in regimes:
                    z = base * scale
                    with torch.no_grad():
                        if ac is None:
                            out = _probe_call(sub, mod, z, mul, att)
                        else:
                            with torch.autocast(device_type=dev.type, dtype=ac):
                                out = _probe_call(sub, mod, z.to(ac), mul, att)
                                z = z.to(ac)
                    mx = float(out.abs().max())
                    s_ok = tuple(out.shape) == tuple(z.shape)
                    d_ok = out.dtype == z.dtype
                    detail.append((label, mx, s_ok, d_ok))
                    if not (mx == 0.0 and s_ok and d_ok):
                        all_zero = False
    finally:
        mod.train(was_training)
    return all_zero, detail


def _max_w(mod):
    return max((float(p.detach().abs().max()) for p in mod.parameters() if p.numel()), default=0.0)


def _digest(params):
    """(sum, max|.|) over every stubbed parameter, in ONE device synchronisation.

    Copied from prefix_cache._digest and extended with the max, because the claim this guards is
    specifically "still at the floor" and a sum alone would not say that. One stack per statistic
    is two reductions and one sync; per-tensor floats would be ~200 syncs at every one of ~230 step
    boundaries.
    """
    if not params:
        return ("0", "0")
    with torch.no_grad():
        tot = torch.stack([p.detach().float().sum() for p in params]).sum()
        mx = torch.stack([p.detach().float().abs().amax() for p in params]).amax()
        both = torch.stack([tot, mx]).tolist()
    return (f"{both[0]:.17g}", f"{both[1]:.17g}")


def scan(head, *, autocast_dtype=None, triangle_multiplicative="torch",
         triangle_attention="torch"):
    """Measure every candidate and say which are dead. No mutation; safe to call on its own.

    DECIDED BY MEASUREMENT, BOTH WAYS. A candidate is dead only if every one of its parameters is
    below FLOOR *and* running it returns exactly 0.0 with the input's shape and dtype in every
    regime probed. Either test alone is known to be wrong: a hardcoded or threshold-only verdict
    goes stale against a different checkpoint, and a FRESH head returns zero from every residual
    branch (zero-init output projections) while its weights are nowhere near the floor.
    """
    stack = getattr(head, "pairformer_stack", None)
    blocks = list(getattr(stack, "blocks", []) or [])
    if not blocks:
        raise SystemExit("STOP: --skip-dead-ops: the head's pairformer_stack has no blocks.")
    # THE KERNEL NAMES ARE VALIDATED, NOT DEFAULTED. An unrecognised string would fall through
    # upstream's if/elif chain to the torch branch, which is precisely the silent mismatch that
    # made the first version of this file certify the wrong implementation.
    mul, att = str(triangle_multiplicative), str(triangle_attention)
    if mul not in KERNELS_MUL or att not in KERNELS_ATT:
        raise SystemExit(
            f"STOP: --skip-dead-ops was given triangle_multiplicative={mul!r} / "
            f"triangle_attention={att!r}; expected one of {KERNELS_MUL} / {KERNELS_ATT}. These "
            f"select which KERNEL the probe runs, and the verdict 'this module emits exactly "
            f"zero' is only evidence about the kernel the run will actually use. An unknown name "
            f"would fall through upstream's dispatch to the torch reference implementation and "
            f"certify a code path nothing executes.")
    # AND PINNED, because _probe_call mirrors PairformerBlock.forward's CALL SIGNATURE -- not its
    # body, but a renamed or added kwarg would silently change what is being probed. The same pin
    # pose_inblock already carries, deliberately reused so there is one value to update.
    try:
        import pose_inblock as _PIB
        _PIB._pin(type(blocks[0]).forward, _PIB.PINNED_BLOCK_SHA, _PIB.PINNED_BLOCK_WHERE)
    except ImportError:                                                  # noqa: BLE001
        pass
    c_z = int(getattr(head, "c_z", 0)) or int(blocks[0].pair_transition.layernorm1.weight.shape[0])
    rows = []
    for bi, blk in enumerate(blocks):
        for sub in CANDIDATES:
            mod = getattr(blk, sub, None)
            if mod is None:
                continue
            if isinstance(mod, _DeadStub):
                raise SystemExit(
                    f"STOP: --skip-dead-ops is already installed on block {bi}'s {sub}. Scanning "
                    f"a tree that already holds stubs would read the stub's zeros as evidence "
                    f"that the module is dead, which is circular. Call uninstall() first.")
            mw = _max_w(mod)
            below = mw < FLOOR
            # THE RUN IS SKIPPED WHEN THE WEIGHTS ARE LIVE, and only then. Probing an alive module
            # is wasted work, and the two tests are an AND, so a live weight is already a verdict.
            if below:
                zero, detail = _probe_zero(mod, sub, c_z, autocast_dtype=autocast_dtype,
                                           triangle_multiplicative=mul,
                                           triangle_attention=att)
            else:
                zero, detail = False, []
            # THE WHOLE SUBTREE, NOT JUST THIS MODULE. The husk shares `_modules`, so a hook on a
            # CHILD (tri_mul_out.layer_norm_in, say) survives in the tree and silently stops
            # firing -- which is the same failure the refusal exists for, one level down. Counting
            # only the module itself bounded the wrong quantity. Caught in review.
            hooks = 0
            for _n, _m in mod.named_modules():
                hooks += (len(getattr(_m, "_forward_hooks", {}) or {})
                          + len(getattr(_m, "_forward_pre_hooks", {}) or {})
                          + len(getattr(_m, "_backward_hooks", {}) or {})
                          + len(getattr(_m, "_backward_pre_hooks", {}) or {}))
            train = [n for n, p in mod.named_parameters() if p.requires_grad]
            rows.append({
                "block": bi, "sub": sub,
                "name": f"pairformer_stack.blocks.{bi}.{sub}",
                "type": type(mod).__name__,
                "max_w": mw, "below_floor": below, "probe_zero": zero, "probe": detail,
                "hooks": hooks, "trainable": train,
                "cubic": sub != "pair_transition",
                "dead": bool(below and zero),
            })
    return rows


def install(model, *, autocast_dtype=None, triangle_multiplicative="torch",
            triangle_attention="torch", loud=True):
    """Replace every measurably dead candidate with a husk. Returns a report, or None if none.

    Every precondition is a HARD STOP, not a downgrade, following phases/prefix_cache.py: a flag
    that declines to engage is indistinguishable from one that engages and is wrong, and epoch time
    is the only signal either way.
    """
    head = getattr(model, "confidence_head", None)
    if head is None:
        raise SystemExit("STOP: --skip-dead-ops: the model has no confidence_head.")
    if _S["on"]:
        raise SystemExit("STOP: --skip-dead-ops is already installed. Installing twice would scan "
                         "a tree that already holds stubs.")
    rows = scan(head, autocast_dtype=autocast_dtype,
                triangle_multiplicative=triangle_multiplicative,
                triangle_attention=triangle_attention)
    dead = [r for r in rows if r["dead"]]

    # ---- 1. REFUSE IF A DEAD MODULE IS TRAINABLE ------------------------------------------------
    #
    # READ FROM requires_grad ON THE TENSORS, not from the --train-only string that was supposed to
    # set it -- the same discipline prefix_cache uses, and for a sharper reason here.
    #
    # WHY THIS IS A REFUSAL AND NOT A SKIP. A stubbed module receives no gradient AT ALL: its .grad
    # stays None rather than becoming an explicit zero. That is a different object than the
    # all-reduced zeros _apply_step builds for a parameter autograd left ungradiented, so the
    # optimiser's parameter list, the flat collective buffer and AdamW's moment state would all
    # change shape relative to a run without the flag. Silently training nothing is worse than not
    # starting.
    bad = [r for r in dead if r["trainable"]]
    if bad:
        lines = "\n".join(
            f"    {r['name']}  max|w| = {r['max_w']:.6e}, output exactly 0.0, "
            f"{len(r['trainable'])} trainable tensor(s)" for r in bad)
        raise SystemExit(
            f"STOP: --skip-dead-ops would stub {len(bad)} module(s) that --train-only / LoRA has "
            f"left TRAINABLE:\n{lines}\n"
            f"These modules emit exactly zero at the shipped weights, so the arm is spending "
            f"capacity on tensors that currently contribute nothing -- but a stubbed module gets "
            f"NO gradient (its .grad stays None rather than becoming zeros), which changes what "
            f"the optimiser and the gradient collective see. The resolution is for the CALLER to "
            f"DROP {'these modules' if len(bad) > 1 else 'this module'} FROM --train-only "
            f"(e.g. remove {bad[0]['name'].split('.')[-1]!r} from the pattern), not for this flag "
            f"to quietly leave them running. If you MEANT to train them, drop --skip-dead-ops "
            f"instead: training a module out of the floor is a legitimate arm and this flag would "
            f"discard it.")

    # ---- 2. REFUSE A MODULE THAT SOMETHING ELSE IS WATCHING OR RANDOMISING ----------------------
    hooked = [r for r in dead if r["hooks"]]
    if hooked:
        raise SystemExit(
            f"STOP: --skip-dead-ops would stub {len(hooked)} module(s) that already carry a "
            f"forward or backward hook, e.g. {[r['name'] for r in hooked[:3]]}. The husk does not "
            f"share hook dictionaries, so whatever registered those hooks would stop being called "
            f"-- silently. Remove the hook or drop the flag.")
    # THERE ARE NONE TODAY, AND THE COMMENT SAYS SO RATHER THAN IMPLYING OTHERWISE. Enumerated
    # over all 20 candidate subtrees of the live head: zero nn.Dropout instances. The only one in
    # a PairformerBlock is `dropout_row.dropout`, a SIBLING of the candidates, and the rate that
    # actually matters (blk.p_drop) is deliberately not checked -- the stub's zeros go through the
    # same dropout_add_rowwise call, so the draw and the scaling are unchanged either way. This
    # loop is therefore a guard against an upstream that ADDS dropout inside one of these
    # modules, not a bound on anything that exists; it is kept for that and is not evidence.
    drops = []
    for r in dead:
        mod = getattr(head.pairformer_stack.blocks[r["block"]], r["sub"])
        for n, m in mod.named_modules():
            if isinstance(m, torch.nn.Dropout) and float(getattr(m, "p", 0.0)) != 0.0:
                drops.append((r["name"], n, float(m.p)))
    if drops:
        raise SystemExit(
            f"STOP: --skip-dead-ops would stub module(s) containing a live dropout: "
            f"{drops[:4]}. Not running them would consume a different number of random draws than "
            f"running them, so every downstream draw in the step would shift and the flag would "
            f"stop being bit-identical -- which is its entire claim.")

    if not dead:
        if loud:
            print("DEAD OPS: nothing to skip -- every one of "
                  f"{len(rows)} candidate module(s) in the confidence head's pairformer stack "
                  f"either holds live weights or did not return exactly zero when run. "
                  f"--skip-dead-ops is a NO-OP on this checkpoint.")
        _S["on"] = False
        _S["report"] = rows
        return None

    # ---- 3. SWAP THEM OUT ----------------------------------------------------------------------
    keys_before = set(head.state_dict().keys())
    names_before = {n for n, _ in head.named_parameters()}
    blocks = head.pairformer_stack.blocks
    # _S["head"] IS SET *BEFORE* THE SWAP, AND THAT IS THE WHOLE POINT. The three guards below
    # call uninstall() and say "Rolled back" -- and uninstall() finds the modules to restore via
    # _S["head"]. Assigning it only in the _S.update() at the end made all three rollbacks SILENT
    # NO-OPS: reproduced in review by forcing the key-set guard to fire, after which all 20 stubs
    # were still installed, the head's state_dict had collapsed from 252 keys to 84, and _S["orig"]
    # had been cleared -- the original modules unreachable and the head unrecoverable. The message
    # was false in all three branches.
    _S["head"] = head
    _S["orig"], _S["stubs"], _S["params"], _S["names"] = {}, {}, [], []
    for r in dead:
        blk = blocks[r["block"]]
        orig = getattr(blk, r["sub"])
        stub = _DeadStub(orig, r["name"])
        # setattr ON THE BLOCK, so nn.Module.__setattr__ swaps the entry in the block's _modules
        # and the husk is what the block's forward (native or mirrored) looks up by attribute.
        setattr(blk, r["sub"], stub)
        # AND TELL pose_inblock's MIRROR TO SKIP THE TRANSPOSES TOO. A stub avoids the attention
        # but not the two `.contiguous()` copies that bracket it -- those exist only to flip z onto
        # the axis tri_att_end attends over, and flipping back is a round trip to nowhere once the
        # branch is a no-op. Each copy is the whole pair representation: 2.50 GiB in bf16 at 2,289
        # tokens. Measured: stub alone still OOMed there at 90.85 GiB allocated.
        #
        # SET ONLY HERE, where the module has just been PROVEN zero by running it, and only for
        # tri_att_end, which is the one bracketed by transposes. pose_inblock reads the flag and
        # never writes it, so the elision cannot outlive the proof. uninstall() clears it.
        if r["sub"] == "tri_att_end":
            blk._verified_skip_tri_att_end = True
        _S["orig"][(r["block"], r["sub"])] = orig
        _S["stubs"][(r["block"], r["sub"])] = stub
        for n, p in orig.named_parameters():
            _S["params"].append(p)
            _S["names"].append(f"{r['name']}.{n}")

    # ---- 4. AND IT MUST ACTUALLY HAVE TAKEN, WITHOUT CHANGING THE TREE -------------------------
    #
    # The key set is the thing phase_k_eval guards the head checkpoint with, so a husk that dropped
    # names would turn a speedup into an unloadable checkpoint one epoch later.
    keys_after = set(head.state_dict().keys())
    if keys_after != keys_before:
        uninstall()
        raise SystemExit(
            f"STOP: --skip-dead-ops changed the head's state-dict key set "
            f"(+{sorted(keys_after - keys_before)[:3]} / -{sorted(keys_before - keys_after)[:3]}). "
            f"The husk is supposed to carry the original's parameters, buffers and children by "
            f"reference precisely so that phase_k_eval's set-equality guard on head tensors still "
            f"passes. Rolled back; do not launch.")
    names_now = {n for n, _ in head.named_parameters()}
    if names_now != names_before:
        uninstall()
        raise SystemExit("STOP: --skip-dead-ops changed the head's named_parameters() set. "
                         "--train-only, LoRA and prefix_cache all select by name. Rolled back.")
    for r in dead:
        if not isinstance(getattr(blocks[r["block"]], r["sub"]), _DeadStub):
            uninstall()
            raise SystemExit(f"STOP: --skip-dead-ops did not take: {r['name']} is not a stub after "
                             f"assignment. Rolled back.")

    # THE DIGEST IS TAKEN OVER THE NAMES IN SORTED ORDER, which is the order step_check() will
    # re-derive them in. A stacked sum over a different ordering can differ in its last bits,
    # and the comparison is exact.
    _S.update({"on": True, "head": head, "calls": 0, "report": rows})
    _S["digest"] = _digest([dict(head.named_parameters())[n] for n in sorted(_S["names"])])

    n_cubic = sum(1 for r in dead if r["cubic"])
    n_cubic_all = sum(1 for r in rows if r["cubic"])
    info = {
        "stubbed": [r["name"] for r in dead],
        "kept": [r["name"] for r in rows if not r["dead"]],
        "max_w": {r["name"]: r["max_w"] for r in rows},
        "n_stubbed": len(dead), "n_candidates": len(rows),
        "n_cubic_stubbed": n_cubic, "n_cubic": n_cubic_all,
        "n_tensors": len(_S["params"]),
        "n_params": int(sum(p.numel() for p in _S["params"])),
        "floor": FLOOR,
        "triangle_multiplicative": str(triangle_multiplicative),
        "triangle_attention": str(triangle_attention),
    }
    if loud:
        print(f"DEAD OPS: {len(dead)} of {len(rows)} candidate module(s) in the confidence head's "
              f"pairformer stack emit EXACTLY ZERO and are not run -- "
              f"{n_cubic} of the {n_cubic_all} O(N^3) pair operations, plus "
              f"{len(dead) - n_cubic} pair transition(s). Each was DECIDED BY MEASUREMENT, not by "
              f"a list: every parameter below {FLOOR:g} AND the module RUN and returning exactly "
              f"0.0 at the input's shape and dtype, at |z| x1 and x1e3, in train and eval mode, "
              f"THROUGH THE KERNELS THIS RUN USES (triangle_multiplicative="
              f"{triangle_multiplicative}, triangle_attention={triangle_attention}) at "
              f"{_PROBE_TOKENS} tokens so the fused attention path is actually entered"
              + (f" and under {str(autocast_dtype).replace('torch.', '')} autocast"
                 if autocast_dtype is not None else "")
              + f". The block's own forward is untouched: it adds a zeros tensor, which is the "
              f"identity on z, so this is BIT-IDENTICAL arithmetic and not an approximation. "
              f"{info['n_tensors']} tensor(s) / {info['n_params']:,} parameter(s) stay in the tree "
              f"-- the husks share them by reference, so the checkpoint key set is unchanged -- "
              f"and are re-checked at every step boundary.")
        for r in rows:
            mark = "SKIPPED" if r["dead"] else "kept   "
            print(f"    {mark}  {r['name']:<46} max|w| = {r['max_w']:.6e}"
                  + ("  (output measured exactly 0.0)" if r["dead"]
                     else ("  (output NON-ZERO when run)" if r["below_floor"] else "")))
    return info


def step_check():
    """Re-assert the premise at a step boundary: the stubbed weights are STILL at the floor.

    Modelled on prefix_cache.step_reset(), which re-reads its frozen prefix at every step boundary
    rather than trusting the flag that was supposed to freeze it. The failure this catches is
    someone re-initialising or loading into a module while the stub is still installed: the arm
    would think it had a live module there, the stub would keep returning zeros, and the only
    symptom would be a loss curve that did not move. A step boundary is immediately after an
    optimiser step, which is the only moment anything could have written those tensors.
    """
    if not _S["on"]:
        return
    # RE-DERIVED FROM THE MODULE TREE EVERY TIME, NOT READ FROM THE LIST CAPTURED AT INSTALL.
    # nn.Module._apply REPLACES the entries in `_parameters` with new Parameter objects whenever
    # the copy is not shallow-compatible -- a device or dtype change is exactly that -- so a held
    # list of objects would, after any later model.to(dev), point at orphaned tensors: the digest
    # would be frozen forever and the requires_grad scan would read stale flags, i.e. this guard
    # would pass unconditionally while guarding nothing. Nothing in the trainer moves the model
    # after install today, so this is closing a latent hole rather than fixing a live bug.
    # (phases/prefix_cache.py holds its prefix list the same way and has the same latent hole.)
    want = set(_S["names"])
    cur = {n: p for n, p in _S["head"].named_parameters() if n in want}
    missing = sorted(want - set(cur))
    if missing:
        raise SystemExit(
            f"STOP: --skip-dead-ops is on and {len(missing)} stubbed parameter(s) have left the "
            f"head's named_parameters(), e.g. {missing[:4]}. The husks carry the originals' "
            f"parameters by reference precisely so that cannot happen, so the module tree has "
            f"been rebuilt or re-wrapped underneath the stubs.")
    names = sorted(want)
    params = [cur[n] for n in names]
    live = [n for n, p in zip(names, params) if p.requires_grad]
    if live:
        raise SystemExit(
            f"STOP: --skip-dead-ops is on and {len(live)} stubbed parameter(s) now require grad, "
            f"e.g. {live[:4]}. A stubbed module is NOT RUN, so it can receive no gradient at all "
            f"and its .grad stays None; whatever just made these trainable is asking for a module "
            f"the run would silently discard. Drop them from --train-only or drop --skip-dead-ops.")
    now = _digest(params)
    if now != _S["digest"]:
        raise SystemExit(
            f"STOP: --skip-dead-ops is on and the weights of a STUBBED module changed value "
            f"(sum/max {_S['digest']} -> {now}). Nothing should write those tensors -- they are "
            f"frozen and their module is not even executed -- so something (an optimiser group, a "
            f"weight load, an EMA, a re-initialisation) has touched them, and every step since has "
            f"been returning zeros from a module that is no longer dead.")
    mx = float(now[1])
    if not (mx < FLOOR):
        raise SystemExit(
            f"STOP: --skip-dead-ops is on and max|w| over the stubbed modules is {mx:.6e}, which "
            f"is no longer below the {FLOOR:g} floor. The digest matched, so this is a floor that "
            f"moved rather than a weight that did -- re-read dead_ops.FLOOR.")


def stats():
    with _LOCK:
        return {"on": bool(_S["on"]), "calls": int(_S["calls"]),
                "stubbed": len(_S["stubs"]), "tensors": len(_S["params"])}


def reset_stats():
    with _LOCK:
        _S["calls"] = 0


def report():
    """The last scan(), whether or not anything was stubbed. For the epoch line and for tests."""
    return list(_S["report"])


def uninstall():
    """Put the original modules back. For tests; the trainer never calls it."""
    head = _S.get("head")
    if head is not None:
        blocks = getattr(getattr(head, "pairformer_stack", None), "blocks", None)
        for (bi, sub), orig in list(_S["orig"].items()):
            cur = getattr(blocks[bi], sub, None) if blocks is not None else None
            if isinstance(cur, _DeadStub):
                # THE MODE COMES BACK TOO. head.train()/eval() while the husk was installed set
                # `training` on the husk and on the shared children but NOT on the detached
                # original, so restoring it blind would reinstate whatever mode it had at
                # install. Transition (pair_transition) genuinely branches on self.training
                # (primitives.py:190-230), so that is a numerical difference, not cosmetic.
                orig.train(bool(cur.training))
                setattr(blocks[bi], sub, orig)
                if sub == "tri_att_end":
                    # CLEARED WITH THE STUB. A flag that outlived uninstall() would silently drop a
                    # live module's whole branch.
                    blocks[bi]._verified_skip_tri_att_end = False
    _S.update({"on": False, "head": None, "orig": {}, "stubs": {}, "params": [], "names": [],
               "digest": None, "calls": 0})
