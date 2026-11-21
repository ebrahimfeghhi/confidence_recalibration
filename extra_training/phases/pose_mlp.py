"""THE NONLINEAR POSE BRANCH: one small MLP, shared first layer, a separate second layer per
injection site.

WHY IT EXISTS. Every pose term in this head is a LINEAR map added into z_pair: a linear map of the
one-hot binned distance (phases/pose_perblock.py and confidence.py), a linear map of the raw
distance (same), and a zero-init linear map of the frame-relative unit direction
(phases/pose_direction.py). Three additive linear terms can represent "far" and can represent
"pointing that way", and they CANNOT represent "far AND pointing that way" -- any interaction
between distance and direction is outside the function class, at every one of the six sites. PAE is
a statement about where j landed relative to i's frame, which is a direction TIMES a length, so the
quantity the head is being asked to predict is itself a product of two inputs it can only add.

WHAT IT ADDS, and nothing else. A two-layer MLP:

    h      = ReLU( W_onehot(one_hot(d_CB)) + W_cont([d_CB/50, dir_ij, dir_ji, d_CA/50]) )   shared
    site k : z_pair = z_pair + (the three existing terms) + out_k(h)                        per site

W_onehot, W_cont and `h` are computed ONCE per pose and reused at all six sites. `out_k` is one
Linear(hidden -> c_z) per site with WEIGHT AND BIAS AT EXACTLY ZERO, so a freshly installed branch
adds exactly zero everywhere and step 0 is bit-identical to the same arm without it -- install()
verifies that rather than asserting it. Nothing existing changes: not one of the three terms, not
its initialisation, not its inputs.

THE SIX SITES, in the order their second layers are indexed:

    0        before pairformer block 0   (pose_direction's pre-stack injection)
    1..n-1   before pairformer blocks 1..n-1   (pose_perblock's wrapped blocks)
    n        before pae_ln   (pose_readout --pose-readout-prenorm)
    n+1      after pae_ln    (pose_readout --pose-readout-post-zero)

so a 4-block stack gives 6, which is what the parameter arithmetic below assumes. The count is
DERIVED FROM THE MODULE TREE, not written down: a stack of a different depth gets a second layer per
depth and says so out loud, for the same reason head_param_names reads the tree instead of carrying
a list.

WHY LAYER 1 IS TWO MATRICES AND NOT ONE. W_onehot(a) + W_cont(b) is exactly one linear layer on
cat([a, b]) -- same function, same parameter count. Written as one layer it would need an
[N, N, n_bins + 8] input tensor, which at the 3,400-token crop cap in float32 is 2.1 GiB
materialised for a single matmul. Split, the one-hot is the tensor pose_perblock already published
and nothing new of that size is built.

GRADIENT LIFE AT ZERO INIT, which is the one place a multi-layer branch differs from the single
layers around it. Layer 2 is zero, so dL/d(layer 1) = 0 on the first step: layer 1 does not move
until layer 2 has. Layer 2 does move immediately, because dL/dW_2 = g (x) h^T and `h` is non-zero
(ReLU of a default-initialised layer on non-zero geometry). So the branch wakes up in two steps
rather than one, and it is NOT gradient-dead -- which a branch with BOTH layers zeroed would be.
pose_readout.py's own docstring makes the same point for the single layers; this is the two-layer
version of it.

WHERE `h` LIVES. It is a trainable-weight ACTIVATION, so it must be recomputed on every forward and
must stay in the autograd graph: it is cached per pose for reuse WITHIN one forward and never across
forwards, and it is never computed under no_grad. The GEOMETRY it is built from is cached -- by
pose_perblock and pose_direction, which already do -- and `h` itself is rebuilt from that cache once
per pose.

Installed on the MODEL INSTANCE, like the other three pose modules: /home/ebrahim/Protenix is a
separate checkout and a change there would not be in this repo's history.
"""
import math

import torch

# 50 ANGSTROM, THE SCALE BOTH DISTANCES ARE DIVIDED BY. Not a tuned constant and not a
# normalisation of the data: it is one fixed divisor, chosen by the request, that puts a protein
# pair distance in roughly [0, 1] next to the direction components, which are already in [-1, 1].
# Zero-init bounds layer 2's OUTPUT at step 0; it does nothing about the SIZE OF THE FIRST UPDATE,
# which is the output error times the input -- the argument pose_direction.py makes at length for
# why it feeds a unit vector rather than a raw displacement. Raw Angstroms of 0..150 next to
# components of +/-1 would give the distance columns of W_cont updates two orders of magnitude
# larger than the direction columns.
#
# THE DIVISION IS ON A COPY. The existing raw-distance layer (linear_no_bias_d_wo_onehot, and
# pose_readout's and pose_perblock's copies of it) keeps receiving unscaled Angstroms from the same
# cached tensor; `dd / DIST_SCALE` allocates, `dd /= DIST_SCALE` would have silently rescaled the
# input of three pretrained layers.
DIST_SCALE = 50.0

HIDDEN = 64

# THE SEED LAYER 1 IS DRAWN FROM. Fixed, and the reason is in install(): this is the only added
# parameter in the project whose value comes from an RNG, and the ranks of a DDP launch never
# exchange parameters. See the comment there before changing it -- a per-rank or per-run value
# would silently make a multi-rank arm an ensemble of first layers.
INIT_SEED = 20261118

# THE CONTINUOUS HALF OF LAYER 1's INPUT, IN THIS ORDER. Named rather than written as 8 so that the
# cat below and the layer width cannot drift apart: a mismatch would be a shape error at install,
# not a silently reordered input.
#
# WHY THE DISTANCE APPEARS TWICE OVER DIFFERENT ATOMS, and why that is not a bug to reconcile --
# confirmed by the user on 2026-11-18. `d_CB` is the distogram representative atom (CB, CA for
# glycine); `d_CA` is the PAE representative atom (the token centre, CA). The one-hot and raw
# distance stay on CB because the pretrained bin edges and the pretrained linear_no_bias_d /
# linear_no_bias_d_wo_onehot weights are defined on those atoms, and moving them would invalidate
# those weights. `d_CA` is CA->CA because it is the length of the SAME arrow as dir(i,j): direction
# times length is j's CA position in i's frame, which is exactly the quantity PAE is defined on,
# and a CB length there would be the wrong length for that arrow. Having both is mildly USEFUL
# rather than merely tolerable: d_CB shorter than d_CA means the two side chains point toward each
# other, so the branch gets a cue about facing that neither distance carries alone -- the kind of
# interaction a nonlinear branch exists to represent. Do not "fix" one to match the other.
_CONT = ("d_CB/50", "dir_ij.x", "dir_ij.y", "dir_ij.z",
         "dir_ji.x", "dir_ji.y", "dir_ji.z", "d_CA/50")
_N_CONT = len(_CONT)

# `h` for the pose currently in flight: [N_token, N_token, HIDDEN], ~1.5 GiB in bf16 at the
# 3,400-token crop cap, which is the price of the branch and is why there is exactly one of it.
#
# CLEARED ON THE WAY OUT OF memory_efficient_forward, including on an exception, exactly as
# pose_perblock clears its one-hot: a pair-sized tensor left reachable between poses is what a
# per-pose leak looks like from the outside. Clearing the dict slot does NOT disturb the backward
# pass -- the blocks captured their own reference (see pose_perblock._prep_blocks), and this drops
# the dict's reference, not theirs.
_CALL = {"h": None}


def _layer1(in_features, out_features, bias):
    """One half of layer 1, with DEFAULT PyTorch initialisation from a FIXED generator.

    The two init calls are copied verbatim from torch.nn.Linear.reset_parameters -- the same
    kaiming_uniform_(a=sqrt(5)) for the weight and the same 1/sqrt(fan_in) uniform for the bias --
    with an explicit generator in place of the global one. Copied rather than re-derived: writing
    the bound out by hand is how "default PyTorch initialisation" quietly becomes something else.
    The generator is seeded per call from one constant, so both halves and every rank get the same
    draw, and nothing else in the process sees its stream move. torch.manual_seed would have been
    the obvious alternative and is wrong here: it reseeds the CUDA generators too, which would
    shift the dropout draws for the whole run and break the bit-identity this flag promises.
    """
    m = torch.nn.Linear(in_features, out_features, bias=bias)
    g = torch.Generator().manual_seed(INIT_SEED)
    with torch.no_grad():
        torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), generator=g)
        if m.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(m.bias, -bound, bound, generator=g)
    return m


def hidden():
    """`h` for the pose in flight, or None. Read by the pae_ln hooks, which run inside the call."""
    return _CALL["h"]


def site_prenorm(head):
    """The second-layer index of the pre-pae_ln site."""
    return len(head.pairformer_stack.blocks)


def site_postnorm(head):
    """The second-layer index of the post-pae_ln site."""
    return len(head.pairformer_stack.blocks) + 1


def install(model, hidden_dim=HIDDEN, loud=True):
    """Add the shared first layer, one zero second layer per site, and compute `h` per pose.

    Installed LAST, after pose_direction, pose_perblock and pose_readout, because it refuses
    unless all three are there: every one of its inputs is a tensor one of them publishes, and
    every one of its outputs lands at a site one of them owns. On its own it would be a silent
    no-op -- the failure --pose-inblock-pae is refused for at phase_k_train.py:1896.
    """
    head = model.confidence_head
    if getattr(head, "_pose_mlp_installed", False):
        raise RuntimeError("STOP: pose_mlp.install ran twice; the branch would be added twice at "
                           "every site through two sets of weights.")

    # EVERY PRODUCER MUST BE PRESENT, AND NAMED IN THE REFUSAL. The branch reads four cached
    # tensors: the one-hot and the raw distance from pose_perblock, the direction and the length
    # from pose_direction. It writes at six sites, two of which only exist with
    # --pose-readout-prenorm AND --pose-readout-post-zero. A missing producer would not crash --
    # the hooks would see None and return the input unchanged -- so an arm would train a
    # four-site branch while its logs claimed six.
    _need = [("--frame-direction", "_pose_direction_installed"),
             ("--pose-perblock", "_pose_perblock_installed"),
             ("--pose-readout", "_pose_readout_installed")]
    _absent = [flag for flag, attr in _need if not getattr(head, attr, False)]
    if _absent:
        raise RuntimeError(
            f"STOP: pose_mlp needs {', '.join(_absent)}. It reads the one-hot and raw distance "
            f"pose_perblock publishes and the direction and length pose_direction publishes, and "
            f"it injects at the sites those modules and pose_readout own. Without them its inputs "
            f"are None and its hooks never fire, so the flag would be a silent no-op.")
    if not (getattr(head, "_pose_readout_prenorm", False)
            and getattr(head, "_pose_readout_post_zero", False)):
        raise RuntimeError(
            "STOP: pose_mlp wants the readout's BOTH sites, so it needs --pose-readout-prenorm "
            "AND --pose-readout-post-zero. With only one of them there are 5 sites, not 6, and "
            "one second layer would be created and never called -- the same failure class as a "
            "--train-only that creates parameters and never trains them.")

    ref = head.linear_no_bias_d.weight
    n_blocks = len(head.pairformer_stack.blocks)
    n_sites = n_blocks + 2

    # THE BIN COUNT COMES FROM THE LAYER THAT ALREADY CONSUMES THE ONE-HOT, not from a literal and
    # not from head.num_bins. linear_no_bias_d.in_features IS the width of the tensor this branch
    # is handed, so a model whose bin count differs fails at install with a shape error rather
    # than at the first matmul of the first pose.
    n_bins = int(head.linear_no_bias_d.in_features)

    # torch.nn.Linear, NOT Protenix's LinearNoBias, FOR LAYER 1 ONLY -- and that is deliberate.
    # protenix.model.modules.primitives.Linear overrides _init_params and initialises its weight
    # with trunc_normal_init_(scale=1.0), which is NOT default PyTorch initialisation. The request
    # names default PyTorch initialisation for both halves of layer 1, so using the project's
    # Linear here would have quietly given a different starting function under a name that claims
    # otherwise. Layer 2 is zeroed, so its class cannot matter; it is nn.Linear too for symmetry.
    l1_onehot = _layer1(n_bins, hidden_dim, bias=False)
    l1_cont = _layer1(_N_CONT, hidden_dim, bias=True)
    head.add_module("pose_mlp_onehot", l1_onehot.to(device=ref.device, dtype=ref.dtype))
    head.add_module("pose_mlp_cont", l1_cont.to(device=ref.device, dtype=ref.dtype))

    # IDENTICAL ON EVERY RANK, VERIFIED HERE. The DDP contract in phase_k_train.py:2063 is that
    # every rank builds an identical replica and only HEAD GRADIENTS are all-reduced -- there is
    # no parameter broadcast anywhere in this project. That held because every module it has ever
    # added is zero-init or a copy of checkpoint weights; layer 1 is the FIRST added parameter
    # whose value comes from an RNG, and phase_k_train never seeds torch's RNG at all (--seed
    # feeds random.Random for data order only). Built from the global generator, the three ranks
    # of a fresh launch would each hold a DIFFERENT layer 1, the all-reduce would sum gradients
    # taken at three different parameter values, and the replicas would stay permanently offset
    # -- an arm training three first layers while reporting one, with rank 0's the only one saved.
    # Nothing downstream could see it: the shapes, the key set, --expect-trainable and the resume
    # guard all match. _layer1 therefore draws from its OWN fixed-seed generator, which also makes
    # a relaunch of the arm reproducible, as every previous arm here was. Re-drawn and compared
    # rather than asserted, because "same seed" is the entire guarantee the ranks rely on.
    # (A fixed seed, not --seed: ranks share --seed today, but nothing enforces that, and the
    # reproducibility of the arm should not depend on it.)
    _again_o, _again_c = _layer1(n_bins, hidden_dim, bias=False), _layer1(_N_CONT, hidden_dim, True)
    if not (torch.equal(_again_o.weight, l1_onehot.weight.cpu().to(_again_o.weight.dtype))
            and torch.equal(_again_c.weight, l1_cont.weight.cpu().to(_again_c.weight.dtype))
            and torch.equal(_again_c.bias, l1_cont.bias.cpu().to(_again_c.bias.dtype))):
        raise RuntimeError(
            "STOP: layer 1's initialisation is not reproducible from its own seed, so two ranks "
            "of a fresh launch would hold different weights and the gradient all-reduce would "
            "average three different models. Nothing downstream can detect that.")

    outs = torch.nn.ModuleList()
    for _ in range(n_sites):
        o = torch.nn.Linear(hidden_dim, head.c_z, bias=True)
        with torch.no_grad():
            o.weight.zero_()
            o.bias.zero_()
        outs.append(o.to(device=ref.device, dtype=ref.dtype))
    head.add_module("pose_mlp_out", outs)

    # VERIFIED, NOT ASSUMED, AND THE BIAS IS WHY. Every other zero-init layer in this project is a
    # LinearNoBias, where zeroing the weight is the whole story. These have a bias, and a bias of
    # 1e-3 would add a constant c_z vector at every pair at every site while every check that only
    # looked at the weight passed. Probed through the module, which is the only form of the check
    # that covers both tensors at once.
    with torch.no_grad():
        probe = torch.randn(4, hidden_dim, device=ref.device, dtype=ref.dtype)
        worst = max(float(o(probe).abs().max()) for o in outs)
    if worst > 0:
        raise RuntimeError(f"STOP: a zero-init second layer returned {worst:.3e}, not zero, so "
                           f"installing the nonlinear branch would change the head at step 0 and "
                           f"nothing measured afterwards would be attributable to it.")

    orig_mef = head.memory_efficient_forward

    def memory_efficient_forward(input_feature_dict, s_trunk, z_pair, pair_mask,
                                 x_pred_rep_coords, **kw):
        # THIS WRAPPER COMPUTES NOTHING. `h` is built at the first injection site, inside
        # pose_direction's wrapper, because that is where the last of its four geometry inputs is
        # published. All this does is guarantee the release, on the normal path and on an
        # exception, for the same reason pose_perblock's wrapper does.
        try:
            return orig_mef(input_feature_dict=input_feature_dict, s_trunk=s_trunk,
                            z_pair=z_pair, pair_mask=pair_mask,
                            x_pred_rep_coords=x_pred_rep_coords, **kw)
        finally:
            _CALL["h"] = None

    head.memory_efficient_forward = memory_efficient_forward
    head._pose_mlp_installed = True
    head._pose_mlp_sites = int(n_sites)
    head._pose_mlp_hidden = int(hidden_dim)

    names = ["pose_mlp_onehot", "pose_mlp_cont", "pose_mlp_out"]
    n_new = sum(p.numel() for n in names for p in getattr(head, n).parameters())
    expect = n_bins * hidden_dim + _N_CONT * hidden_dim + hidden_dim + n_sites * (
        hidden_dim * head.c_z + head.c_z)
    if n_new != expect:
        raise RuntimeError(f"STOP: the branch has {n_new:,} parameters, not the "
                           f"{expect:,} its own shapes predict.")
    info = {"params": n_new, "sites": n_sites, "hidden": hidden_dim, "n_bins": n_bins,
            "n_cont": _N_CONT, "c_z": int(head.c_z), "tensors": names}
    if loud:
        print(f"POSE MLP (nonlinear branch): layer 1 SHARED across all {n_sites} site(s) -- "
              f"pose_mlp_onehot Linear({n_bins} -> {hidden_dim}, no bias) + pose_mlp_cont "
              f"Linear({_N_CONT} -> {hidden_dim}) through ReLU, default PyTorch init; layer 2 is "
              f"{n_sites} SEPARATE pose_mlp_out.k Linear({hidden_dim} -> {head.c_z}) with weight "
              f"AND bias at exactly zero, verified. {n_new:,} new parameter(s) across "
              f"{len(list(head.pose_mlp_out.parameters())) + 3} tensor(s). Sites: 0 = pre-stack, "
              f"1-{n_blocks - 1} = pairformer blocks, {site_prenorm(head)} = before pae_ln, "
              f"{site_postnorm(head)} = after pae_ln. The {_N_CONT} continuous inputs are "
              f"d_CB/{DIST_SCALE:g}, dir(i,j), dir(j,i), d_CA/{DIST_SCALE:g}; the one-hot is the "
              f"tensor pose_perblock already published. STEP 0 IS BIT-IDENTICAL to the same arm "
              f"without this branch. A --train-only must match `pose_mlp` to reach these.")
        # PER-RANK EVIDENCE IN run.log, not just a claim in a comment. Layer 1 is the only added
        # parameter drawn from an RNG and the ranks never exchange parameters, so the one thing a
        # reader may need to confirm weeks later is that all three ranks printed the SAME digest.
        print(f"  layer 1 init: seed {INIT_SEED}, sum "
              f"{float(head.pose_mlp_onehot.weight.double().abs().sum()):.9f} / "
              f"{float(head.pose_mlp_cont.weight.double().abs().sum()):.9f} -- IDENTICAL ON EVERY "
              f"RANK by construction (fixed generator), verified reproducible at install. There "
              f"is no parameter broadcast in this trainer, so a per-rank init would have made "
              f"the arm an ensemble of first layers.")
    return info

def _geometry(head):
    """(onehot, dist, dir, len) for the pose in flight, with every absence named."""
    import pose_direction as _pd
    import pose_perblock as _ppb
    oh, dd = _ppb._CALL.get("onehot"), _ppb._CALL.get("dist")
    dr, ln = _pd._CALL.get("dir"), _pd._CALL.get("len")
    missing = [n for n, t in (("one-hot distance (pose_perblock)", oh),
                              ("raw distance (pose_perblock)", dd),
                              ("direction (pose_direction)", dr),
                              ("length (pose_direction)", ln)) if t is None]
    if missing:
        # RAISES RATHER THAN SKIPPING. install() already proved every producer is installed, so
        # reaching here means the ORDER broke, not the configuration: the branch would contribute
        # zero at every site for the rest of the run while the arm's logs and its args.json both
        # claimed a six-site nonlinear branch. That is the one failure this project pays for
        # repeatedly -- a flag that reports an architecture it is not training.
        raise RuntimeError(
            f"STOP: pose_mlp reached its injection point without {', '.join(missing)}. Every one "
            f"of those is published before the pre-stack site by a module install() verified is "
            f"present, so this means memory_efficient_forward was bypassed or the install order "
            f"changed; the branch would be silently absent.")
    # A SHAPE CHECK, AND ONLY A SHAPE CHECK. It catches a cache from a unit of a different size
    # -- the hooks firing on a call they were not written for -- and it CANNOT catch two poses of
    # the SAME unit being crossed, because those have identical shapes. What covers that is
    # upstream and per sample: pose_direction._delta rebuilds the representative atoms from the
    # stashed coordinates and refuses unless they equal the tensor memory_efficient_forward was
    # handed, which runs before `h` is built. Said plainly here because the version of this
    # comment that claimed "one pose's geometry would pair with another's" promised a guarantee
    # this check does not give.
    if dd.shape != oh.shape[:-1] or dr.shape[:-1] != oh.shape[:-1] \
            or ln.shape[:-1] != oh.shape[:-1]:
        raise RuntimeError(
            f"STOP: pose_mlp's inputs disagree about shape -- one-hot {tuple(oh.shape)}, raw "
            f"distance {tuple(dd.shape)}, direction {tuple(dr.shape)}, length "
            f"{tuple(ln.shape)}. The branch is being fed geometry from a different unit.")
    return oh, dd, dr, ln


def prestack(head, z):
    """Build `h` for this pose, cache it, and add site 0's term to z. Returns the new z.

    Called from pose_direction's memory_efficient_forward wrapper, which is where the last of the
    four geometry inputs has just been published.
    """
    oh, dd, dr, ln = _geometry(head)

    # FLOAT32 GEOMETRY IN, z's DTYPE OUT, BY AUTOCAST AND NOT BY HAND -- exactly how the existing
    # direction term does it. Everything cached is float32 because the bin comparison and the
    # frame basis are not reliable in bf16 (see the autocast-disabled blocks in pose_perblock and
    # pose_direction). These two matmuls are OUTSIDE those blocks, so under the caller's autocast
    # they run in bf16 and `h` comes out bf16: 1.5 GiB at the 3,400-token cap instead of 3.0.
    #
    # NOT UNDER no_grad, AND NOT DETACHED. `h` is a function of trainable weights and is the only
    # path by which layer 1 ever receives a gradient, so it must stay in the graph. The GEOMETRY
    # carries no gradient of its own -- it is built from predicted coordinates, and these four
    # tensors are passed through with exactly the treatment the three existing terms give the same
    # coordinates, neither detached here nor anywhere upstream.
    #
    # NO PAIR MASK AND NO CHUNKING, because the terms this sits beside apply neither: block 0's
    # injection in confidence.py, pose_perblock's per-block adds and pose_readout's hooks are all
    # unmasked full-pair adds, and the chunk_size the stack takes reaches the triangle operations,
    # not the injections. Masking here and nowhere else would make this term the odd one out.
    cont = torch.cat([(dd / DIST_SCALE).unsqueeze(-1),
                      dr,
                      # THE REVERSE DIRECTION IS THE CACHE TRANSPOSED OVER ITS TWO TOKEN
                      # DIMENSIONS, so reverse[i, j] IS forward[j, i] by construction and no
                      # geometry is recomputed. Negated-forward would be wrong: dir(j, i) is j's
                      # frame looking at i, which is a different rotation, not -dir(i, j).
                      dr.transpose(-3, -2),
                      ln / DIST_SCALE], dim=-1)
    h = torch.relu(head.pose_mlp_onehot(oh) + head.pose_mlp_cont(cont))
    # THE FLOAT32 COPY IS DROPPED HERE, AND THAT IS NOT COSMETIC. `cont` is [N, N, 8] float32 --
    # 370 MiB at the 3,400-token crop cap. Autograd must keep SOME version of it until backward,
    # because dL/dW_cont is the output gradient times this input; under autocast what it saves is
    # the BF16 cast F.linear made, 185 MiB. Releasing this reference lets the float32 original go
    # at once instead of being pinned beside the bf16 copy for the whole of the stack's forward
    # and backward. Not needed for the one-hot: pose_perblock's cache holds that one anyway, and
    # its existing per-block projections already pin it the same way.
    del cont
    _CALL["h"] = h
    return add_site(head, 0, z, h)


def add_site(head, k, z, h):
    """z + site k's second layer applied to `h`, in z's dtype. `h` is PASSED IN, never read here.

    WHY `h` IS AN ARGUMENT. Sites 1..n-1 are inside pairformer blocks that run under activation
    checkpointing, where a block's forward is RE-RUN during backward. A site that re-read the
    module-level cache would see None -- the outer wrapper's `finally` has long since cleared it --
    or, worse, a LATER pose's hidden tensor, pairing one pose's activations with another's
    gradient. pose_perblock._prep_blocks captures this tensor into a local on exactly the same
    contract as the one-hot and the direction, and that local is held by the closure, which the
    blocks list holds, which the checkpoint graph holds until its backward is done.
    """
    if h is None:
        raise RuntimeError(
            f"STOP: pose_mlp site {k} was reached with no hidden tensor. It is built at the "
            f"pre-stack site, which runs before every other site, so a None here means the branch "
            f"would be silently absent from this site while the arm reports otherwise.")
    mod = head.pose_mlp_out[k]
    w, b = mod.weight, mod.bias
    # THE WEIGHT IS CAST TO `h`, NEVER `h` TO THE WEIGHT. Two of the six sites are the pae_ln
    # hooks, and confidence.py runs the whole readout inside `with autocast(enabled=False)` on an
    # upcast float32 z_pair. Under AMP `h` is bf16 and these weights are float32, and with autocast
    # off torch does not reconcile that: the module call raises "expected scalar type Float but
    # found BFloat16" at the first pose of the first step. Casting `h` instead would allocate a
    # second float32 copy of it -- 3.0 GiB at the 3,400-token crop cap -- where casting the weight
    # is 8 KiB. F.linear rather than the module because the module is what applies the dtype that
    # fails. Zero-init survives the cast exactly: 0.0 in any float dtype is 0.0.
    if h.dtype != w.dtype:
        w, b = w.to(h.dtype), b.to(h.dtype)
    return z + torch.nn.functional.linear(h, w, b).to(z.dtype)
