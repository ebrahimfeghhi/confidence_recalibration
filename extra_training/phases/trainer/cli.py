"""Every command-line flag the trainer takes, and nothing else.

WHY THIS IS ITS OWN FILE. phase_k_train.py is one 8,380-line main(); 1,259 of those lines are
argparse. Pulling them out is the one extraction that cannot change behaviour -- an ArgumentParser
is built, not run -- and it removes 15% of the file before anything riskier is touched.

The flags and their help text are moved VERBATIM. The help strings carry the reasoning behind
defaults that were measured rather than guessed, and that reasoning is the documentation; it is not
summarised here.

phases/trainer/test_cli_equivalence.py asserts this parser produces a byte-identical Namespace to
the original for the live arm's real argv, so the move is verified rather than assumed.
"""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_parser():
    """The trainer's parser. Returns argparse.ArgumentParser, unparsed."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=f"{ROOT}/training/fold")
    ap.add_argument("--labels", default="", help="default <fold>/labels")
    ap.add_argument("--out", default=f"{ROOT}/training/ft3", help="checkpoints and trajectory")
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--indices", default=f"{ROOT}/training/prep/job_indices.csv")
    ap.add_argument("--bio-dir", default=f"{ROOT}/training/prep/job_bioassembly")
    ap.add_argument("--mmcif-dir", default=f"{ROOT}/training/prep/job_cifs")
    ap.add_argument("--msa-root", default=f"{ROOT}/building_msa/work/mmcif_msa")
    ap.add_argument("--seq-index", default=f"{ROOT}/building_msa/work/pdb_seqs/seq_to_pdb_index.json")
    ap.add_argument("--crop-size", type=int, default=3400)
    ap.add_argument("--alpha-pae", type=float, default=1.0,
                    help="FT3 wants 1. At Protenix's shipped 0.0 the PAE term is multiplied by zero "
                         "and ipTM -- the only number this project is judged on -- cannot move.")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Protenix ships TWO: configs_base.py has 0.0018 and their own "
                         "finetune_demo.sh passes 0.001. FT3 is a finetune, so 0.001 is the default "
                         "here. Neither was tuned for training 4.3%% of the model at batch 256, so "
                         "--lr-probe exists to measure it rather than inherit it.")
    ap.add_argument("--warmup-steps", type=int, default=20,
                    help="SHORT BECAUSE THE RUN IS SHORT. Protenix's finetune_demo.sh warms up over "
                         "2,000 steps of a 100,000-step run (2%%). At batch 256 three epochs of our "
                         "cache is 292 optimiser steps, so 2,000 would be 685%% of the run and the "
                         "learning rate would never reach its set value. 20 is 6.8%%.")
    ap.add_argument("--lr-steps", default="",
                    help="A STAIRCASE schedule, EPOCH:LR,EPOCH:LR -- e.g. "
                         "20:1e-3,40:1e-4 holds --lr until epoch 20, then 1e-3 until epoch "
                         "40, then 1e-4. TAKES PRECEDENCE over --lr-final and the decay "
                         "window/shape, which interpolate continuously and so cannot "
                         "express a hold-then-drop. Warmup still applies to the first "
                         "steps. Empty (default) leaves every existing arm bit-identical.")
    ap.add_argument("--save-init", default="",
                    help="write the head to this path BEFORE any optimiser step, then exit. This "
                         "is what step-zero parity needs and nothing else produced it: every "
                         "other head save happens after at least one step, and both pose "
                         "injections are zero-init, so a head saved after one step is no longer "
                         "bit-identical to the un-injected one. With it, phase_k_eval can be "
                         "handed a genuine in-block checkpoint whose PAE and ipTM MUST reproduce "
                         "the shipped head exactly -- which tests the evaluator's install path, "
                         "its key-set guard and the zero-init claim in one comparison, on the "
                         "real CLI rather than in a test harness. Refuses under WORLD_SIZE > 1: "
                         "it exits the process, and exiting one rank of three parks the others "
                         "on a collective.")
    ap.add_argument("--rank-loss", action="store_true",
                    help="PAIRWISE RANKING LOSS on chain_pair_iptm, the statistic selection "
                         "actually uses. The FT3 PAE loss is a per-pair accuracy objective; "
                         "selection is an argmax of ipTM over an interface's poses. Those are "
                         "different functionals of the same logits and they move independently: "
                         "measured mean within-interface Spearman(ipTM, DockQ) is 0.196 for the "
                         "shipped head, 0.271 for the pose-share arm at epoch 50, and 0.078 for "
                         "the full head at epoch 30 -- BELOW untrained. Training PAE taught the "
                         "big arms to rank worse. This trains the ordering instead. Needs DockQ "
                         "labels (--rank-labels) and --pae-scored-pairs, which is what names the "
                         "interface whose ipTM is read.")
    ap.add_argument("--alpha-rank", type=float, default=1.0,
                    help="weight on the ranking term. It is ADDED to the PAE term, so "
                         "--alpha-pae 0 --rank-loss trains the ordering alone and "
                         "--alpha-pae 1 --rank-loss trains both. NOT TUNED: the two terms are "
                         "different quantities (a PAE loss near 2.3 against a softplus ranking "
                         "loss whose tied value is ln2 = 0.693), so this ratio is a guess until a "
                         "sweep says otherwise.")
    ap.add_argument("--rank-pool", choices=("max", "mean"), default="max",
                    help="HOW A POSE'S ipTM IS POOLED OVER ANCHOR ROWS, in BOTH pass A and the "
                         "pass-B term, so the quantity trained is the quantity scored. max is "
                         "shipped ipTM. mean measured +0.0140 +- 0.0034 (t=4.06) better than the "
                         # ESCAPED. argparse runs every help string through `% params`, so a bare
                         # "67.7% of" is read as the conversion "% o" and `--help` died with
                         # "TypeError: %o format: an integer is required, not dict" -- the whole
                         # help, not just this entry. Present at 72148ac3 and found while adding
                         # the two flags below, which nobody could otherwise have read.
                         "max at the ORACLE on 738 winnable interfaces (67.7%% of oracle headroom "
                         "against 58.6%%) and -0.0130 +- 0.0048 (t=-2.68) WORSE when swapped onto "
                         "the current trained head -- better if trained for, worse if swapped in. "
                         "It also gives gradient to every row rather than to the argmax alone, "
                         "which removes the moving-target problem the max has.")
    ap.add_argument("--rel-labels", default="",
                    help="directory of <pdb>.npz true-iTM labels from "
                         "run_logging/build_itm_labels.py, keyed (seed, sample, asym_x, asym_y) "
                         "-> g_mean, g_max. REQUIRED by --rank-form rel, which regresses on them "
                         "instead of on DockQ.")
    ap.add_argument("--rel-standardize", action="store_true",
                    help="ABLATION. Also divide each interface's centred prediction and label by "
                         "its own std (floored), which makes the loss invariant to shrinking the "
                         "scores so collapse becomes impossible rather than merely penalised. OFF "
                         "by default because it equalises interfaces whose outcomes are NOT equal: "
                         "measured, corr(within-interface std of the true iTM, DockQ range) = "
                         "+0.745, so wide-spread interfaces are exactly where a wrong pick is "
                         "expensive, and the narrow ones have frac acceptable median 1.00.")
    ap.add_argument("--rank-form", choices=("linear", "softplus", "expected",
                                            "spearman", "distance", "spearman_distance",
                                            "rel"),
                    default="linear",
                    help="SHAPE of the ranking term. linear and softplus are PAIRWISE; expected "
                         "is not.\n"
                         "rel is MEAN-CENTRED REGRESSION ON THE TRUE iTM and is not a ranking "
                         "loss at all: L = (1/P) sum_i [(s_i - mean_s) - (g_i - mean_g)]^2 over "
                         "an interface's poses, with g from --rel-labels. It exists because the "
                         "head's failure is measured to be positional, not collapse: on d32kqp "
                         "epoch 66, corr(interface MEAN ipTM, interface MEAN DockQ) = +0.866 "
                         "while the median WITHIN-interface Spearman is +0.317 against +0.690 for "
                         "the true iTM. Subtracting each interface's own mean from both sides "
                         "removes the across-complex signal the head already has, so the only "
                         "thing left to earn is the within-complex ordering selection uses. Its "
                         "coefficients sum to EXACTLY zero, so it cannot move an interface's "
                         "level and its accumulated gradient cancels by construction -- set "
                         "--alpha-rank from gradient NORMS via --grad-probe, never from loss "
                         "values.\n"
                         "expected is L = -sum_i softmax(ipTM/tau)_i * DockQ_i over every pose of "
                         "the interface: the expected DockQ of the pose the model would SELECT. "
                         "It exists because a pairwise loss spends gradient in proportion to how "
                         "many pairs a pose appears in, which is the same for rank 1 and rank 90, "
                         "while deployment takes the argmax and nothing else -- so ordering the "
                         "bottom of the list is work that cannot pay. h7msge is the evidence: "
                         "held-out pairwise loss reached a run best and Spearman went +0.075 over "
                         "shipped, while selected DockQ stayed flat-to-down and frac acceptable "
                         "never beat shipped once in 37 epochs. The softmax is a differentiable "
                         "argmax, so tau sets how much mass sits on the current leaders; it has "
                         "NO PAIRS, so --rank-weight-by-gap and the cross-seed pair filter have "
                         "nothing to act on and are reported as inapplicable rather than silently "
                         "appearing to work. Its floor is -max(DockQ) for the group, not 0.\n"
                         "linear (default, what every arm so far ran) is "
                         "-(s_i - s_j)(dq_i - dq_j)/sum|ddq|. Its coefficients come from the "
                         "LABELS ALONE, so it needs no scoring pass -- and that convenience IS "
                         "its defect: a gradient that ignores the current scores is constant in "
                         "them, so the loss has no stopping point and cannot tell an "
                         "already-correct pair from an inverted one. Scaling all scores by lambda "
                         "scales the loss by lambda, so WIDENING THE SPREAD PAYS WITHOUT LIMIT, "
                         "and at the real gap scale spreading is worth ~71x more than reordering. "
                         "Measured in ntwjfp: spread 0.0114 -> 0.0371 over 8 epochs while held-out "
                         "rank loss went +0.00018 -> +0.00288 -- the training objective improved "
                         "while held-out ORDERING got worse.\n"
                         "softplus is (2 tau/W) sum w_ij softplus(-m_ij/tau), whose per-pair "
                         "gradient weight sigmoid(-m/tau) decays to 0 once a pair is comfortably "
                         "correct and caps at 1 when badly inverted. Spreading then buys a bounded "
                         "amount and stops. Costs the scoring pass (--rank-form softplus implies "
                         "one), which is exactly the price of having somewhere to stop.")
    ap.add_argument("--rank-cross-seed", action="store_true",
                    help="RANK ACROSS SEEDS, not just within one. Deployment selects argmax of "
                         "ipTM over ALL poses of an interface regardless of seed, so cross-seed "
                         "ordering IS the objective and within-seed ordering is a proxy. Measured "
                         "variance split: ipTM is 82.3%% between-seed while DockQ is 54.6%% "
                         "pose-level.\n"
                         "This regroups the epoch so that ALL seeds of a structure land in one "
                         "optimiser step, which makes the ranking group the whole interface (50 "
                         "poses, ~1,225 cross-seed pairs) rather than a 3-seed sample. Measured "
                         "agreement between a subsample's gradient direction and the full group's, "
                         "as cosine over shared poses (400 interfaces, 8 draws each): 3 seeds mean "
                         "0.856 with p10 0.544, 5 seeds mean 0.925 with p10 0.823, 10 seeds exact. "
                         "Costs nothing in batch size -- a step is ~20 units either way -- so it "
                         "trades structures per step for an exact ranking signal.")
    ap.add_argument("--rank-structs-per-step", type=int, default=2, metavar="S",
                    help="with --rank-cross-seed, how many STRUCTURES share an optimiser step. "
                         "S structures x all their seeds x --nsample poses is the step. Default 2 "
                         "= 2 x 10 x 5 = 100 poses, matching the ~100 the current schedule already "
                         "uses. S x seeds need not divide the rank count; the short rank rotates "
                         "and the totals are checked at startup rather than being allowed to "
                         "deadlock.")
    ap.add_argument("--train-only", default="",
                    help="SELECTIVE FINETUNING: comma-separated REGEXES matched against parameter "
                         "names inside the confidence head; only matching tensors train and "
                         "everything else is frozen. Empty (default) trains the whole head.\n"
                         "POSE-FOCUSED ADAPTATION is the motivating case: "
                         "'linear_no_bias_d,linear_no_bias_d_wo_onehot,linear_no_bias_pae' is "
                         "13,312 parameters -- 0.07%% of the head -- and is the COMPLETE pose "
                         "pathway plus the PAE readout. Pose enters the head exactly once, as "
                         "cdist over representative atoms feeding those two distance embeddings "
                         "(both, not either), and leaves as linear_no_bias_pae(pae_ln(z_pair)). "
                         "Freezing the four pairformer blocks keeps the pretrained structural "
                         "representation and adapts only how geometry ENTERS and how PAE is READ "
                         "OUT -- a constraint on which computation may change, not merely on how "
                         "much. Add 'pae_ln' to make the readout's scale and shift learnable too "
                         "(+256 parameters).")
    ap.add_argument("--accum", type=int, default=256,
                    help="POSES PER OPTIMISER STEP -- the effective batch. The micro-batch is forced "
                         "to 1: token counts run 263-3,400 across our structures, so poses cannot be "
                         "stacked into one tensor without padding the head's pair representations. "
                         "256 at the user's direction. For reference, Protenix's own finetune runs "
                         "one structure per step with iters_to_accumulate=1 and scales by world_size; "
                         "256 is AF3's paper figure, for full-model training across many devices. At "
                         "256 an epoch of our 25,000 poses is 97 steps.")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--eval-every-epoch", action="store_true",
                    help="score the validation metric at the end of every epoch. THE METRIC IS THE "
                         "RESULT, not the loss: mean DockQ and frac-acceptable of the pose selected "
                         "by max pair_iptm, against the 0.5775 / 0.7283 baseline at 100 seeds.")
    ap.add_argument("--eval-pool", "--train-eval-pool", dest="train_eval_pool", default="",
                    help="score the head on THIS pool of TRAINING interfaces at epoch boundaries, "
                         "in addition to (or instead of) validation. The question `is the model "
                         "learning at all` is about the training set, and --eval-every-epoch only "
                         "knows how to score validation. Uses the run's own --fold and --seeds, so "
                         "it scores exactly the poses it trained on. Sharded over every card the "
                         "run was given, with the other ranks parked at a barrier, same as the "
                         "validation path.")
    ap.add_argument("--eval-pool-every", "--train-eval-every", dest="train_eval_every", type=int, default=2,
                    help="epochs between training-set evaluations. Ignored without "
                         "--train-eval-pool.")
    ap.add_argument("--extra-eval", action="append", default=[], metavar="SPEC",
                    help="AN ADDITIONAL eval pass at its own cadence, repeatable. The "
                         "trainer historically had exactly ONE: --eval-pool and "
                         "--train-eval-pool are argparse aliases for a single dest, so "
                         "\"cropped validation every 2 epochs AND uncropped validation and "
                         "in-sample every 6\" was not expressible and section 12 recorded "
                         "in-sample as simply unmeasured for the uncropped arm.\n"
                         "SPEC is comma-separated key=value with keys: name (required, "
                         "names metric_<name>_ep<N>.json and <name>_eval_shards_ep<N>), "
                         "pool (required), fold, seed_base, budget, every (required), "
                         "crop_tokens. Example:\n"
                         "  --extra-eval 'name=valfull,pool=training/valfull/pool.csv,"
                         "fold=validation/fold,seed_base=400000,budget=30,every=6'\n"
                         "Each pass costs a full sharded evaluation, so every=1 on three "
                         "specs triples that cost; the cadence is the knob.")
    ap.add_argument("--eval-pool-fold", "--train-eval-fold", dest="train_eval_fold", default="",
                    help="score --train-eval-pool against THIS fold instead of --fold. The point "
                         "is to watch a held-out signal every few epochs without paying for the "
                         "whole validation set: a 10-structure validation pool is ~1,000 units "
                         "against 5,300, so minutes rather than ~25. Defaults to --fold, which is "
                         "the in-sample training-pool behaviour. Pair it with "
                         "--train-eval-seed-base, because a different fold was folded at "
                         "different seeds, and with --train-eval-budget, which otherwise defaults "
                         "to the count of THIS run's --seeds and would silently score a tenth of "
                         "a 100-seed fold.")
    ap.add_argument("--eval-pool-seed-base", "--train-eval-seed-base", dest="train_eval_seed_base", type=int, default=-1,
                    help="the first seed of --train-eval-fold. -1 (default) uses this run's own "
                         "--seeds, which is right only when the pool lives in --fold.")
    ap.add_argument("--eval-pool-budget", "--train-eval-budget", dest="train_eval_budget", type=int, default=0,
                    help="seeds per interface for the training-set metric. 0 derives it from "
                         "--seeds, which is the only honest default: the pool has no more depth "
                         "than the run trained on.")
    ap.add_argument("--prefetch-workers", type=int, default=1,
                    help="HOW MANY crop workers run concurrently. One is not enough: epochs 7 "
                         "and 8 hit a 100%% prefetch hit rate and still reported 11-16 min per "
                         "epoch WAITING on the worker, because a single thread needs ~0.7 s per "
                         "unit against ~0.84 s of GPU and the GIL eats that margin. The crop did "
                         "not disappear, it moved from the main thread into the wait. Workers "
                         "take units round-robin, so results arrive out of order and the holding "
                         "buffer is sized for it. HISTORY, READ THIS BEFORE RAISING IT: 1083ed10 "
                         "set 3 everywhere and was reverted in 7033430b because ggfhj5 lost epoch "
                         "12 to what looked like a deadlock -- every rank at 0%% GPU in "
                         "futex_wait, nine minutes with no trajectory write. The mechanism was "
                         "found later and FIXED: _pf_pending evicted unconsumed results, and "
                         "since a worker emits each unit exactly once, the next take waited the "
                         "whole 120 s budget for something that could never arrive, then the unit "
                         "after it did the same. Evictions are now recorded and such a take "
                         "returns immediately; a take issued after every worker has finished "
                         "returns immediately too. run_logging/prefetch_deadlock_repro.py is the "
                         "CPU-only regression test and must stay at 0 stalls for the fixed path.")
    ap.add_argument("--prefetch-units", type=int, default=0,
                    help="prepare this many units ahead on a worker thread, overlapping the CPU "
                         "crop with the GPU's work on the previous unit. 0 disables it. Measured "
                         "on epoch 5 of the full set: crop_slice is 23.0 min of a 73.1 min epoch "
                         "with the GPU idle throughout, and GPU busy time is only ~20 min of 73. "
                         "The worker never touches CUDA -- it hands back host tensors and the main "
                         "thread moves them -- so only one thread ever issues CUDA work.")
    ap.add_argument("--trunk-memmap", action="store_true",
                    help="MAP z_trunk INSTEAD OF READING IT INTO A HEAP BUFFER. The trunk npz's "
                         "members are STORED, not deflated, and z_trunk.npy is the first member at "
                         "zip header_offset 0 -- verified on three real trunks -- so np.memmap can "
                         "address it in place and the host-to-device copy reads page cache "
                         "directly instead of a buffer np.load just filled. That removes one pass "
                         "over ~243 MB per unit, ~1.1 TB per rank per epoch, worth an estimated "
                         "~6 min of a 232 min epoch. z_trunk is ~99%% of the bytes and is the only "
                         "member mapped; the rest are at or under ~1.5 MB where the extra syscalls "
                         "cost more than the copy. OFF BY DEFAULT AND UNTESTED ON A GPU: "
                         "phases/test_trunk_memmap.py gates the two paths bit-identical on CPU "
                         "over real trunks, bf16 compared through a uint16 view, and any member "
                         "that cannot be mapped falls back to np.load silently. NOTE it moves the "
                         "disk read out of trunk_npz_read and into trunk_h2d, because a mapping is "
                         "lazy -- read their SUM when comparing against a run without it.")
    ap.add_argument("--trunk-cache-gb", type=float, default=0.0,
                    help="HOLD A STEP'S UNCROPPED TRUNKS AS HOST ARRAYS, up to this many GiB per "
                         "rank; 0 is off. The cross-seed objective runs TWO passes over every "
                         "step's units -- a no_grad scoring pass whose scores the relative-iTM "
                         "coefficients need, then the backward pass -- so each unit's ~243 MB was "
                         "read and copied to the device TWICE per step (4,600 load_unit calls for "
                         "2,300 units, rank 0, epoch 3 of mqird6). Caching the HOST arrays for the "
                         "span of one step halves the reads, and --prefetch-workers threads warm "
                         "them while the GPU runs, which puts most of the rest behind compute. "
                         "Estimated 10-20 min of a 232 min epoch. KEYED BY TRUNK PATH AND NEVER "
                         "POPPED, unlike the cropped prefetch: a pop queue walked by both passes "
                         "is the dsswrj bug that cost ~3 h an epoch. Purely advisory -- every miss "
                         "falls back to the ordinary read, so there is no wait and nothing to "
                         "deadlock on -- and REFUSES rather than evicts when full, because every "
                         "entry in a step is wanted twice. Ignored when cropping is on (the crop "
                         "prefetch already covers that path) or off a cuda device (host arrays "
                         "would be ALIASED, not copied). 4.0 fits about half a 30-unit step's "
                         "~7.3 GB, so expect a hit rate near 75%% there and read `full` for how "
                         "many were turned away; the epoch line's rate is 50%% when the warmers "
                         "contributed nothing and 100%% when every unit was warmed before pass A. "
                         "CHANGES NO NUMBER the run produces, and is UNTESTED ON A GPU.")
    ap.add_argument("--prefix-cache-gb", type=float, default=0.0,
                    help="HOLD THE FROZEN PREFIX OF THE CONFIDENCE HEAD FOR ONE STEP, up to this "
                         "many GiB of HOST memory per rank; 0 is off. Different cache from "
                         "--trunk-cache-gb: that one holds the INPUTS (z_trunk et al) so they are "
                         "read once instead of twice, this one holds an ACTIVATION from partway "
                         "THROUGH the head so pass B does not recompute it. Under "
                         "--pose-inblock --lean-single the deepest point at which z is a function "
                         "of frozen parameters alone is INSIDE the injected block, after "
                         "tri_mul_in and before the pose enters -- the same boundary the injection "
                         "is placed at, seen from the other side -- because --train-only freezes "
                         "blocks 0-2 AND that block's two tri-muls. So a hit skips three whole "
                         "blocks plus two of the last block's five pair operations, which is most "
                         "of a forward: gpu_forward was 69.7 min over 23,000 calls on mqird6 epoch "
                         "6, ~34.9 per pass, and the prefix is ~87%% of it. Expect ~25 min off a "
                         "222 min epoch, net of the D2H on the fill and the H2D on the hit. "
                         "SCOPE IS ONE OPTIMISER STEP: pass A fills, pass B hits, the step "
                         "boundary drops it -- the epoch's poses are distinct, so nothing is "
                         "reused across steps and a persistent cache would be ~6 TB an epoch "
                         "against a 3,072 GiB quota. ON THE HOST, because z is [N, N, 256] bf16 = "
                         "174 MB at the median 583 tokens and a 50-pose step is ~8.7 GB typical / "
                         "~26 GB at the p90, which no card has spare. CAPPED IN BYTES WITH THE "
                         "RESERVATION TAKEN BEFORE THE COPY, so the cap bounds RSS and not just "
                         "the dict; a pose that does not fit RECOMPUTES. Purely advisory -- every "
                         "miss recomputes the prefix, so NO NUMBER the run produces depends on a "
                         "hit. REFUSES to enable unless --pose-inblock is on, --lean-single is on, "
                         "every prefix tensor reads requires_grad=False at runtime and every "
                         "dropout rate in the prefix blocks is 0.0. Validated on CPU by "
                         "run_logging/test_prefix_cache.py; UNTESTED ON A GPU.")
    ap.add_argument("--skip-dead-ops", action="store_true",
                    help="DO NOT RUN CONFIDENCE-HEAD MODULES WHOSE OUTPUT IS EXACTLY ZERO. The "
                         "shipped checkpoint carries pairformer modules whose EVERY tensor sits in "
                         "a band 4.904e-37..4.907e-37 -- a fixed exponent with only the low "
                         "mantissa bits varying, not a trained weight and not a denormal (fp32's "
                         "smallest normal is 1.175e-38). Two such factors multiplied underflow to "
                         "zero, so the module emits EXACTLY 0.0: measured on the shipped head, "
                         "NINE of the SIXTEEN O(N^3) pair operations (block 0 entirely, blocks "
                         "1-2's two triangle attentions, block 3's tri_att_end) plus block 1's "
                         "pair_transition. Every one is consumed as z + module(z), so adding its "
                         "zero is the identity on z and NOT RUNNING IT IS BIT-IDENTICAL "
                         "ARITHMETIC, not an approximation -- the module object is replaced by a "
                         "husk returning zeros_like(z) and the block's own forward is untouched, "
                         "so there is no third mirror of PairformerBlock.forward to go stale. "
                         "DECIDED BY MEASUREMENT AT EVERY INSTALL, never from a list: a module is "
                         "skipped only if every parameter is below 1e-30 AND running it returns "
                         "exactly 0.0 at the input's shape and dtype, at |z| x1 and x1e3, in "
                         "train and eval mode and under bf16 autocast. A hardcoded list would go "
                         "stale against a different checkpoint, and a threshold alone would call "
                         "a FRESH head fully dead (its residual branches are zero-init). The "
                         "husks share the originals' parameters by reference, so the head's "
                         "state-dict key set, --train-only's name matching and the evaluator's "
                         "set-equality guard are all unchanged, and the weights are re-checked "
                         "against the floor at every step boundary. REFUSES rather than degrading "
                         "if a module it would skip is in the TRAINABLE set -- a stubbed module "
                         "gets no gradient at all (.grad stays None rather than becoming zeros), "
                         "which changes what the optimiser and the gradient collective see, so "
                         "the resolution is to drop it from --train-only. NOTE this arm's "
                         "--train-only includes block 3's tri_att_end, which IS dead, so the "
                         "refusal fires until that pattern changes. COMPLEMENTARY to "
                         "--prefix-cache-gb, not redundant with it: the cache removes the frozen "
                         "prefix from pass B, this removes half of pass A, which the cache leaves "
                         "running in full. Validated on CPU by run_logging/test_dead_ops.py; "
                         "UNTESTED ON A GPU.")
    ap.add_argument("--val-loss-every", type=int, default=0,
                    help="EPOCHS BETWEEN HELD-OUT PAE PASSES; 0 is off. This is the monitor the "
                         "run had no way to compute: measured 2026-10-20, the lr 1e-2 anchor's "
                         "training PAE fell 38%% from epoch 1 to 142 while its val10 PAE ROSE 51%% "
                         "(2.8290 -> 4.2639), and nothing in the run could see it -- the DockQ "
                         "metric stayed inside its noise band the whole time because ipTM barely "
                         "orders poses within an interface (rho 0.20 even for the shipped head). "
                         "At 500 poses it costs ~3 min against a 37-min full-set epoch.")
    ap.add_argument("--val-loss-fold", default="", help="fold root for the held-out pass")
    ap.add_argument("--val-loss-labels", default="", help="defaults to <val-loss-fold>/labels")
    ap.add_argument("--val-loss-selection", default="", help="selection csv; needs a `tokens` column")
    ap.add_argument("--val-loss-pool", default="", help="pool csv for the PAE mask on the held-out side")
    ap.add_argument("--val-loss-preflight", default="", help="that fold's alignment preflight marker")
    ap.add_argument("--val-loss-seeds", default="", help="e.g. 400000-400009")
    ap.add_argument("--val-loss-prep", default="",
                    help="prep root holding job_indices.csv, job_bioassembly and job_cifs for that "
                         "fold. The dataset config defaults to the TRAINING prep, and pointing it "
                         "at the wrong one fails loudly -- 10 structures with no datapoint -- "
                         "rather than measuring the wrong molecules.")
    ap.add_argument("--val-loss-card", default="",
                    help="card for the held-out pass. Empty shares the trainer's; it loads a second "
                         "464M-parameter model, so the free-memory floor below applies.")
    ap.add_argument("--val-loss-shard-retries", type=int, default=2,
                    help="how many times to re-run a FAILED held-out PAE shard before voiding "
                         "the epoch's number. The shards read a bundle already on disk and the "
                         "weights cannot move while they run, so a re-run measures exactly the "
                         "same model as its siblings; 0 restores the old void-on-any-failure "
                         "behaviour. tv6hue epoch 4 lost two of three shards to a transient "
                         "CUDA context-creation failure and voided a point worth 2.1 min of work.")
    ap.add_argument("--val-loss-uncropped", action="store_true",
                    help="run the held-out PAE pass on the FULL structure instead of the training "
                         "crop. OFF BY DEFAULT and it must stay off mid-comparison. WHY ANYONE "
                         "WANTS IT: evaluation should match the deployment setting, and the DockQ "
                         "eval is already uncropped -- so today the two held-out passes see "
                         "DIFFERENT INPUTS for the same poses, which is the only reason they "
                         "cannot share one forward. Uncropping this one is the prerequisite for "
                         "merging them, worth ~4-5 min of a ~10 min eval phase. WHAT IT COSTS, "
                         "and it is not small: the pass inherits the run's crop precisely so the "
                         "held-out number is comparable to the TRAINING curve it is read against, "
                         "which is the only thing it is for. Uncropped, train-vs-val PAE stops "
                         "being apples to apples and the generalisation-gap readings (copy's gap "
                         "wider than share's by +0.0067 +- 0.0017 over 12 matched epochs) are no "
                         "longer computable the same way. It also puts a SEAM in the val PAE "
                         "curve at whichever epoch it lands, and makes that arm's val PAE "
                         "incomparable to every arm measured before it -- so turn it on at the "
                         "start of a fresh comparison, never inside a running one.")
    ap.add_argument("--val-loss-emit-iptm", action="store_true",
                    help="have the held-out PAE pass also write chain_pair_iptm rows, for "
                         "run_logging/dockq_from_iptm.py to rebuild the DockQ metric from. Only "
                         "useful together with --val-loss-uncropped: dockq_from_iptm --compare "
                         "checks PER-POSE ipTM against the eval's own rows, and a cropped pass "
                         "sees different inputs, so every pose would differ for a legitimate "
                         "reason and the join error it exists to catch would be buried. With the "
                         "inputs identical, ANY ipTM difference is the join.")
    ap.add_argument("--emit-iptm-rows", default="",
                    help="ALSO write chain_pair_iptm per (interface, seed, sample) to this csv. "
                         "EXPERIMENTAL and off by default. The held-out PAE pass and the DockQ "
                         "eval forward the SAME 7,950 poses through the SAME head every epoch -- "
                         "verified identical as sets of (pdb, seed, sample) -- and ipTM is derived "
                         "from the PAE logits the loss pass already has, so the second forward is "
                         "redundant in principle. This emits the rows that would let the DockQ "
                         "metric be computed from the loss pass, WITHOUT removing the eval, so "
                         "the two can be compared on real data before anything is deleted. "
                         "THE JOIN IS THE RISK, not the arithmetic: phase_k_eval joins poses to "
                         "DockQ by COORDINATE IDENTITY against the _sample_<k>.cif files, and a "
                         "naive (iface, seed, sample) key join is only equivalent if the sample "
                         "ordering agrees. Measured on j583i8 epoch 1 it does, on all 13,800 rows "
                         "against validation/fold/results.csv -- but that is one pool on one day, "
                         "which is exactly why the eval keeps its own self-check and why nothing "
                         "here replaces it yet.")
    ap.add_argument("--manifest-shard", type=int, default=-1,
                    help="TAKE ONLY THIS SHARD of --manifest-selection, INTERLEAVED (i::n), with "
                         "--manifest-nshards. -1, the default, takes everything and is what every "
                         "existing caller gets, so adding this changes no behaviour unless asked "
                         "for. INTERLEAVED and not contiguous because the selection csv is "
                         "ordered and contiguous blocks hand one shard all the large complexes -- "
                         "phase_k_eval_sharded.py's own note, and the reason its speedup would "
                         "otherwise be set by the slowest block. Exists for the held-out PAE "
                         "monitor, which was a SINGLE-card child while two cards sat at a "
                         "barrier: 12.8 min of a 46.0 min epoch cycle in j583i8.")
    ap.add_argument("--manifest-nshards", type=int, default=0,
                    help="how many shards --manifest-shard is one of. 0 disables sharding.")
    ap.add_argument("--pae-only", action="store_true",
                    help="train the PAE term ALONE: pLDDT, PDE and resolved are stubbed out, not "
                         "computed, not trained. ipTM derives entirely from PAE, and at the shipped "
                         "weights PAE is only ~30%% of the loss (pLDDT 42%%, PDE 27%%), so most of "
                         "the gradient goes to outputs the metric never reads. THE RISK IS REAL "
                         "THOUGH: all four outputs come off one shared confidence head, so the "
                         "auxiliary terms may be regularising the representation PAE reads from. "
                         "Composes with --pae-scored-pairs, which narrows WHICH PAE pairs count.")
    ap.add_argument("--pae-scored-pairs", default="",
                    help="a pool CSV. The PAE term is then computed ONLY on token pairs that cross "
                         "a chain pair this pool scores -- not on intra-chain pairs, and not on "
                         "chain pairs nothing grades (H-L above all). chain_pair_iptm reads exactly "
                         "one cross-block per interface, while the unmasked term averages over "
                         "every pair in the crop, so most of the PAE gradient is spent on blocks "
                         "the metric never looks at. Masks the loss, not the crop: same tokens, "
                         "same forward pass, no extra compute.")
    ap.add_argument("--pae-scored-roles", default="",
                    help="chain_roles.csv for --pae-scored-pairs. Defaults to <fold>/chain_roles.csv. "
                         "Only the SET of scored pairs matters here, and that set is unchanged by "
                         "swapping the two antibody chains, so the chemically-identical-chain "
                         "mislabelling that makes roles unusable for SCORING does not reach it.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="500000-500009",
                    help="the seed range the manifest expects per structure, inclusive.")
    ap.add_argument("--manifest-selection",
                    default=f"{ROOT}/train_val_splits/cache_train_structures.csv",
                    help="THE FROZEN POPULATION. The manifest is this times --seeds, not whatever "
                         "label files happen to exist; a label that was never written must be an "
                         "error, not a silently smaller training set.")
    ap.add_argument("--manifest-skip", default="9mx1",
                    help="comma-separated pdb ids the manifest must NOT expect, each a deliberate "
                         "documented exclusion. 9mx1 is 2,941 residues against Protenix's "
                         "2,560-token limit, so it cannot be folded at all.")
    ap.add_argument("--max-nonfinite", type=int, default=10,
                    help="how many non-finite-gradient steps to tolerate before stopping. Each one "
                         "drops its gradients and takes no step; past this the data or the learning "
                         "rate is wrong and continuing trains on noise.")
    ap.add_argument("--max-rss-gb", type=float, default=300.0,
                    help="HARD CAP ON THIS PROCESS'S OWN MEMORY, checked before every item is added "
                         "to the cache. The user's constraint is that our processes do not exceed "
                         "300 GB, and --preload-min-free-gb answers a different question: it looks "
                         "at MemAvailable, so on an idle box it would let one run take 700 GB. Past "
                         "this the cache stops growing and evicts; the run never fails for it.")
    ap.add_argument("--preload-min-free-gb", type=float, default=200.0,
                    help="FLOOR ON FREE RAM, checked before every item is added to the cache -- not "
                         "once at startup. The cache GROWS to ~253 GB over the first epoch (500 "
                         "items at a measured 331-886 MB, avg 518), so a startup check would pass "
                         "and the box could still be driven out of memory later, taking the fold "
                         "workers with it. When MemAvailable falls below this the cache stops "
                         "growing and evicts instead; it never fails the run.")
    ap.add_argument("--grad-check-every", type=int, default=25,
                    help="re-assert that no gradient has appeared outside the confidence head")
    ap.add_argument("--save-every", type=int, default=250, help="optimiser steps per checkpoint")
    ap.add_argument("--mem-log", action="store_true",
                    help="print live GPU memory after every pose, with the token count. For "
                         "diagnosing whether memory ACCUMULATES (a retained reference) or merely "
                         "peaks (a capacity limit) -- three peak-reducing changes did not move when "
                         "the run died, which is not what a capacity limit looks like.")
    ap.add_argument("--blocks-per-ckpt", type=int, default=1,
                    help="ACTIVATION CHECKPOINTING over the head's pairformer blocks. Chunking "
                         "shrinks the SCRATCH inside one operation; this discards the activations "
                         "kept BETWEEN blocks and recomputes them in the backward pass, which is "
                         "the other half of the memory and the larger half. 1 checkpoints every "
                         "block (most saved, about 30%% more compute). Protenix supports it "
                         "natively on PairformerStack and it engages only when gradients are "
                         "enabled, so the evaluator is unaffected. 0 disables it.")
    ap.add_argument("--ckpt-token-threshold", type=int, default=0,
                    help="SIZE-CONDITIONAL ACTIVATION CHECKPOINTING. Units at or below this many "
                         "tokens run with checkpointing OFF; larger units keep "
                         "--blocks-per-ckpt. 0 (default) disables the rule, so every unit uses "
                         "--blocks-per-ckpt and nothing changes.\n"
                         "WHY IT IS SAFE TO VARY PER UNIT. The peak is a PER-POSE property, not a "
                         "per-step one: _one_pose is a function, its frame is destroyed on "
                         "return, and every large tensor is freed before the next pose allocates "
                         "(the pose-lifetime fix). So a small unit running without checkpointing "
                         "cannot add to the peak that a large unit sets -- it only has to stay "
                         "UNDER it. The run's peak is 87.71 GB allocated on cards of 95.6 GB with "
                         "the largest unit at 1543 tokens, and that is the number a threshold "
                         "must not raise.\n"
                         "WHY IT PAYS. blocks_per_ckpt=1 recomputes every one of the head's four "
                         "pairformer blocks in backward, about 30%% more compute, and it is "
                         "applied to the 400-token units because the 1543-token ones need it. "
                         "gpu_backward is 70.3 min of a 143.9 min epoch.\n"
                         "SET IT FROM run_logging/bench_ckpt_by_size.py, NOT BY GUESSING. The "
                         "benchmark measures peak memory and CUDA-event time per unit at each "
                         "setting and reports the largest token count whose unchecked peak stays "
                         "under the checkpointed peak, with a margin. A threshold chosen by "
                         "arithmetic would be an OOM 40 minutes into an epoch.\n"
                         "IT CHANGES NO MATHEMATICS. Checkpointing trades compute for memory and "
                         "is gradient-exact; test_perblock_ckpt_equiv.py already gates "
                         "blocks_per_ckpt=1 against off at 0.000e+00 on every parameter, which is "
                         "exactly the equivalence this relies on.")
    ap.add_argument("--lean-readouts", action="store_true",
                    help="STOP COMPUTING THE THREE READOUTS --pae-only ALREADY THROWS AWAY. "
                         "--pae-only stubs plddt_loss, pde_loss and resolved_loss and skips the "
                         "all-atom lddt, so none of the three is trained -- but the HEAD still "
                         "builds pde_pred, plddt_pred and resolved_pred on every pose, and this "
                         "trainer puts all four into `pred`. Because nothing in the loss reads "
                         "them, backward never traverses those branches, and a branch backward "
                         "does not walk is not freed by backward either: their activations live "
                         "until _one_pose returns, through the window where the peak is set.\n"
                         "WHAT IT FREES, from the live checkpoint's shapes (c_z 256, c_s 384, "
                         "b_plddt 50), all fp32: the pde chain is N_tok^2 * 2304 B and the "
                         "plddt/resolved chain is N_atom * 84 KB, dominated by the "
                         "plddt_weight[atom_to_tokatom_idx] gather at 76.8 KB PER ATOM. That is "
                         "~1.3 GB at a median 617-token unit and ~6.5 GB at the 1543-token worst "
                         "case, against the 1.8 GB of headroom section 11 measured. ARITHMETIC, "
                         "not a measurement -- run_logging/bench_ckpt_by_size.py measures it.\n"
                         "WHY IT PAIRS WITH --ckpt-token-threshold: the threshold is limited by "
                         "headroom, so freeing headroom raises the threshold, and the two "
                         "compound.\n"
                         "IT CHANGES NO MATHEMATICS ON THE PAE PATH. phases/lean_readouts.py "
                         "mirrors ConfidenceHead.memory_efficient_forward with the pae lines "
                         "verbatim and is PINNED to that function's sha256, so a Protenix upgrade "
                         "hard-stops instead of silently running a stale copy. "
                         "run_logging/test_lean_readouts.py checks the pae logits are "
                         "BIT-identical, that the dropped work is not merely unused but never "
                         "called, and that the plddt placeholder still survives the indexing "
                         "loss.py:1772 does to it.\n"
                         "REQUIRES --pae-only, and is NOT inherited by the --loss-only val child, "
                         "which builds its own model in its own process.")
    ap.add_argument("--lean-single", action="store_true",
                    help="STOP COMPUTING THE PAIRFORMER'S SINGLE TRACK, which --lean-readouts left "
                         "running. PairformerBlock.forward updates z through five ops that are "
                         "functions of z ALONE, then updates s through attention_pair_bias and "
                         "single_transition, both of which read z -- so z NEVER DEPENDS ON s. "
                         "Under --pae-only the readout is linear_no_bias_pae(pae_ln(z_pair)), "
                         "which reads z only, and s feeds nothing but the pLDDT and resolved heads "
                         "--lean-readouts already stubs. The single track was therefore computed "
                         "in all 4 blocks for every pose of every unit of every epoch and thrown "
                         "away; lean_readouts.py's own comment says 's_single is now unused by "
                         "anything that is kept' and then keeps computing it.\n"
                         "HOW: PairformerBlock gates it on `if self.c_s > 0:` in both __init__ and "
                         "forward, so setting c_s=0 on a BUILT block stops the calls without "
                         "touching the module tree -- no renamed keys, no dropped state-dict "
                         "entries, and phase_k_eval's set-equality check on head tensors still "
                         "passes. Both that forward and PairformerStack.forward are PINNED to a "
                         "sha256 so a Protenix upgrade hard-stops rather than silently removing "
                         "the guard or the one-directional dependency this rests on.\n"
                         "WHAT IT IS WORTH, measured on CPU at the live configuration rather than "
                         "argued: 10,053,120 parameters (50.90%% of the head) stop being nominally "
                         "trainable, and with the readout parameters nothing reads either, "
                         "10,551,552 of 19,752,270 (53.42%%) are dropped from AdamW and from the "
                         "gradient all-reduce -- ~84 MB of moment state and ~42 MB of the 79 MB "
                         "flat collective buffer per step. BUT THE COMPUTE SAVING IS SMALL AND "
                         "THAT IS THE HONEST HEADLINE: the pair track is O(N^3) and the single "
                         "track's dominant term is O(N^2), so saved-activation bytes measured 4.0%% "
                         "of a block at 128 tokens, 2.9%% at 256 and 2.5%% at 384, and the "
                         "forward+backward time difference was BELOW CPU MEASUREMENT NOISE "
                         "(-4.7%% to +9.2%% across four token counts). Expect ~1-2%% of the epoch, "
                         "not a large win. It is here because it is free and provably dead, not "
                         "because it is big.\n"
                         "NUMERICS: the PAE logits are unchanged. "
                         "run_logging/test_lean_single.py gates z BIT-IDENTICAL with the single "
                         "track on and off, and gates that the single pathway's parameters "
                         "receive no gradient EITHER WAY -- which is the proof that they were "
                         "already dead rather than being killed by this flag.\n"
                         "REQUIRES --lean-readouts AND --pae-only. UNTESTED ON A GPU.")
    ap.add_argument("--pose-inblock", action="store_true",
                    help="INJECT THE POSE INSIDE THE LAST PAIRFORMER BLOCK, after tri_mul_in and "
                         "before tri_att_start, with BOTH the 39-bin one-hot distance term and the "
                         "raw-distance term -- the same pair of projections the native pre-stack "
                         "injection uses (confidence.py:297-305), shapes (c_z, num_bins) and "
                         "(c_z, 1).\n"
                         "THE PLACEMENT IS THE WHOLE POINT AND IS WHY THIS IS NOT A MODE OF "
                         "--pose-perblock. Autograd starts building at the first op whose "
                         "parameters or inputs require grad. With blocks 0-2 frozen and block 3's "
                         "two triangle multiplications frozen, injecting at the block's INPUT -- "
                         "which is where --pose-perblock injects -- makes the z entering "
                         "tri_mul_out require grad, brings BOTH tri-muls back into the backward, "
                         "and evaporates the saving SILENTLY: no error, no warning, just a slower "
                         "epoch. After tri_mul_in is the earliest point at which the frozen "
                         "tri-muls are already behind us.\n"
                         "ZERO-INIT AND VERIFIED EXACTLY ZERO, by probing the modules rather than "
                         "trusting the initialiser, so step 0 is BIT-IDENTICAL to the un-injected "
                         "head and any later divergence is attributable.\n"
                         "IT IS A MIRROR AND IT IS PINNED. Injecting between two ops inside "
                         "PairformerBlock.forward means reimplementing that function for one "
                         "block; the sha256 is pinned so a Protenix reorder is a hard stop rather "
                         "than an injection that silently lands somewhere else. Only the "
                         "inplace_safe=False branch is mirrored, which is the one this trainer "
                         "takes.\n"
                         "run_logging/test_pose_inblock_graph.py asserts the backward graph starts "
                         "where intended AND carries a negative control that moves the injection "
                         "to the block input and checks the tri-muls DO get gradient -- without "
                         "that control the assertion proves nothing. UNTESTED ON A GPU.")
    ap.add_argument("--pose-inblock-block", type=int, default=-1,
                    help="WHICH BLOCK --pose-inblock injects into; -1 means the LAST one, which is "
                         "the block the freeze leaves trainable. Any other value is for probing "
                         "and must be paired with a --train-only that leaves that block's "
                         "attention and transition trainable, or the injection trains against "
                         "frozen downstream weights.")
    ap.add_argument("--pose-inblock-pae", action="store_true",
                    help="ALSO INJECT THE POSE AT THE READOUT, after pae_ln: "
                         "linear_no_bias_pae(pae_ln(z) + pose_d(one_hot(d)) + pose_d_wo(d)). "
                         "AFTER the layer norm, which is a deliberate choice with a consequence: "
                         "an UNNORMALISED quantity is added to a NORMALISED one and nothing "
                         "downstream rescales it, since linear_no_bias_pae is the next and last "
                         "op. So the injection's effective influence is set by its initialisation "
                         "and by the optimiser, not by the norm. Injecting BEFORE the norm would "
                         "have bounded it. Implemented as a forward HOOK on pae_ln rather than a "
                         "wrapper module, because wrapping would rename pae_ln.weight to "
                         "pae_ln.ln.weight and break phase_k_eval's set-equality check on head "
                         "tensors. Zero-init and verified zero. Requires --pose-inblock.")
    ap.add_argument("--alpha-confidence", type=float, default=1.0,
                    help="weight on the four confidence losses. Protenix ships 1e-4 because in full "
                         "AF3-style training these terms sit beside the diffusion and distogram "
                         "losses and must not perturb them. FT3 freezes the trunk and optimises "
                         "NOTHING ELSE, so that weight is a constant on the whole objective -- it "
                         "moves no minimum, but it is not harmless either: it pushed per-parameter "
                         "gradients to ~1.8e-07, at or below AdamW's eps of 1e-8, so eps dominated "
                         "the denominator and damped the typical step ~12x (MEASURED: a step at "
                         "lr 1e-3 moved parameters by 8.1e-05). Default 1.0, at the user's "
                         "direction on 2026-10-16, so lr means what it says.")
    ap.add_argument("--clip-norm", type=float, default=25.0,
                    help="gradient-norm clip. THIS SCALES WITH --alpha-confidence and must be set "
                         "with it. At the old 1e-4 weight the norm was 7.9e-04 against a threshold "
                         "of 1.0, so clipping never fired and was decorative. At weight 1.0 the "
                         "same gradients are norm 7.9 (median) and 11.8 (max, measured over this "
                         "run's first 8 steps), so a threshold of 1.0 would fire on EVERY step and "
                         "silently turn this into normalised gradient descent. 25 is ~3x the "
                         "measured median: it catches a genuine spike and never binds on a normal "
                         "step. It is a judgement call, derived from measurement rather than picked "
                         "in advance, and it should be revisited if the gradient scale changes.")
    ap.add_argument("--eval-cards", default="",
                    help="cards to SHARD the per-epoch evaluation over, comma separated. Default: "
                         "every card this run was given, since the other ranks are parked at a "
                         "barrier throughout and their cards would otherwise idle for the whole "
                         "~73 min pass. One card falls back to the unsharded evaluator.")
    ap.add_argument("--dist-timeout-min", type=float, default=240.0,
                    help="COLLECTIVE TIMEOUT IN MINUTES, and it must exceed the per-epoch "
                         "evaluation. torch's default is 30 min; the evaluation at --eval-budget "
                         "100 covers 5,300 validation units and takes ~73 min, measured. Rank 0 "
                         "evaluates while the other ranks wait at a barrier, so on the default the "
                         "barrier would time out and kill the run at the FIRST epoch boundary -- "
                         "after hours of correct training, with the failure looking like a network "
                         "fault rather than a timeout someone chose. 240 min leaves room for an "
                         "evaluation that runs long under contention.")
    ap.add_argument("--dist-backend", default="gloo", choices=("gloo", "nccl"),
                    help="the collective backend for multi-GPU. DEFAULT gloo, NOT nccl, because "
                         "NCCL is broken on this box: a 20-line two-rank all_reduce on cards 1 and "
                         "3 fails with `CUDA error: an illegal memory access was encountered`, with "
                         "P2P enabled and disabled alike, on NCCL 2.26.2+cuda12.2 under a torch "
                         "built for 12.8 and driver 595.99.02. It is not this script -- the same "
                         "failure reproduces with no model loaded. gloo is a CPU backend, so the "
                         "gradients are staged through host memory: measured 60 ms per 79 MB "
                         "all-reduce, about 6 s per epoch over ~98 steps, against an epoch of "
                         "hours. Switch to nccl if the driver/NCCL pairing is ever fixed.")
    ap.add_argument("--nsample", type=int, default=5,
                    help="poses per unit, as the fold wrote them. Only asserted under multi-GPU, "
                         "where a unit with a different count would shift one rank's optimiser step "
                         "onto a different pose and deadlock the next all-reduce.")
    ap.add_argument("--allow-code-drift", action="store_true",
                    help="accept a label whose fingerprint differs ONLY in the `code` field, with "
                         "cif, bio, idx, trunk, loss and schema all matching. The data provenance "
                         "is what determines whether a label is CORRECT; the code hash additionally "
                         "says it is REPRODUCIBLE, and it covers whole files, so editing a docstring "
                         "in phase_j_build_cache.py invalidates all 5,000 labels. OFF by default: a "
                         "drift in any other field, or several fields, is still fatal, and every "
                         "accepted drift is printed with both hashes.")
    ap.add_argument("--eval-script", default=f"{ROOT}/phases/phase_k_eval.py")
    ap.add_argument("--eval-budget", type=int, default=100,
                    help="seeds per interface for the epoch-end metric. 100 is the full budget and "
                         "the one the 0.5775 / 0.7283 baseline is quoted at; a pass is ~73 min, measured.")
    ap.add_argument("--resume", default="",
                    help="a resume bundle written at an epoch boundary (resume_ep<N>.pt). Restores "
                         "the head's weights AND AdamW's moment estimates AND the step counter, so "
                         "the warmup position and the per-parameter adaptive scales continue rather "
                         "than restarting -- a weights-only warm start throws both away. Training "
                         "picks up at epoch N+1, which reproduces the right data order because the "
                         "per-epoch reshuffle is random.Random(seed + ep) and depends on nothing "
                         "else. REFUSES a bundle whose world size, seed, accum or nsample differ "
                         "from this run's: all four change which poses land in which batch, so the "
                         "optimiser state would belong to a different problem.")
    ap.add_argument("--minimal-items", action="store_true",
                    help="ACCEPTED AND IGNORED -- this is now the only path, and --alignment-preflight is required whether or not it is passed. Kept so the launch scripts that pass it keep working. MEASURED "
                         "2026-10-17 by recording every key access during a real "
                         "calculate_losses call on four structures spanning 128-1,581 tokens and "
                         "2-5 chains: the loss reads nine feature keys and two label keys, and all "
                         "eleven are already in the cached label npz -- feat_ds and label_dict "
                         "contribute nothing. phase_k_minimal_item_equivalence.py then showed the "
                         "two paths give bit-identical losses on 50 of 50 comparisons. The item is "
                         "the run's memory: ~518 MB each, 164 per rank, so 82-95 GB per rank and "
                         "~264 GB across three, which is what breached the box's free-RAM budget, "
                         "stopped the cache growing at --max-rss-gb and degraded the pace from 2.8 "
                         "to 3.4 min/step as structures began to be rebuilt. The label npz is "
                         "0.11 MiB. REQUIRES --alignment-preflight, because check_alignment is the "
                         "one other thing the item was used for and it cannot run without one.")
    ap.add_argument("--alignment-preflight", default="",
                    help="path to the marker phase_k_alignment_preflight.py writes once it has "
                         "confirmed, for every structure in the manifest, that the trunk's features "
                         "agree atom for atom with the dataset's. --minimal-items refuses without "
                         "it: that check is the one failure that would pair a pose with another "
                         "molecule's label, and skipping it silently is not a trade worth making.")
    ap.add_argument("--resume-allow-drift", action="store_true",
                    help="BLANKET override: accept EVERY setting that differs from the bundle. "
                         "Prefer --resume-override, which names the fields and so still refuses "
                         "the difference you did not expect. Both record identically.")
    ap.add_argument("--resume-override", default="",
                    help="comma-separated GUARD FIELDS this resume may change, e.g. "
                         "'lr,warmup_steps'. A difference in any field NOT named here is still "
                         "refused, which is the whole point: a blanket --resume-allow-drift also "
                         "accepts the change you did not intend. Naming a field that does not "
                         "actually differ is an error too -- a stale override silently weakens "
                         "the guard on the next resume that reuses the command line.")
    ap.add_argument("--resume-note", default="",
                    help="why this resume deviates, recorded beside the accepted fields in "
                         "resume_drift.json and carried forward in every bundle this run writes. "
                         "Required when --resume-override or --resume-allow-drift accepts "
                         "anything: an unexplained deviation is the one a later reader cannot "
                         "tell from a mistake.")
    ap.add_argument("--require-replay", default="",
                    help="path to a replay result that must say PASS before training starts. "
                         "Cached training is only FT3 if the cache reproduces it; see "
                         "phase_j_replay.py. Pass 'skip' to train without the gate and say why.")

    return ap


def parse(argv=None):
    """Parse and resolve. Use this, not build_parser(), so derived defaults are applied.

    `--labels` defaults to <fold>/labels, which argparse cannot express because it depends on
    another flag. The original does this on the line after parse_args; it lives here so that
    "the config" is one function and args.json's values are reproducible from argv alone.
    """
    a = build_parser().parse_args(argv)
    a.labels = a.labels or os.path.join(a.fold, "labels")
    return a
