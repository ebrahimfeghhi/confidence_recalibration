"""Quality-weighted asymmetric regression of mean-row ipTM ONTO DockQ. Reference implementation.

See docs/itm_quality_weighted_loss.md for the rationale and the measurements it rests on.

THE LOSS, for one interface's P candidate poses:

    s_p = mean over anchor rows of t_i(predicted PAE logits)      # the model's score
    y_p = dockq_p                                                 # the target, detached
    w_p = sigmoid((dockq_p - 0.23) / width)                       # how acceptable the pose is

    up   = sum_p   w_p   * relu(y_p - s_p)^2  /  max(sum_p   w_p,   1)   # acceptable, underrated
    down = sum_p (1-w_p) * relu(s_p - y_p)^2  /  max(sum_p (1-w_p), 1)   # unacceptable, overrated
    L    = up + down

THE TARGET IS DockQ ITSELF, not the true iTM, and that is a correction to the first version of
this file. The first version regressed onto g_p = the iTM computed from the TRUE error bins, on
the reasoning that s_p is a bounded structured function of the logits and so could not be asked to
output arbitrary values. THAT REASONING WAS WRONG: s_p = mean of 1/(1+(err/d0)^2) and the model
controls err, so driving the predicted errors far below d0 sends s_p toward 1 and far above it
sends s_p toward 0. s_p spans (0,1) for ANY structure. Nothing constrained it to a sub-range.

The g_p target cost exactly what that false constraint bought, and it was measured: g_p and DockQ
agree at only within-group Kendall tau +0.48, in 40 of 415 mixed groups (9.6%) an UNACCEPTABLE
pose had a higher true iTM than every acceptable one, and since s == g gives L == 0 the loss was
perfectly satisfied by the wrong selection there -- it PREFERRED it. That capped the method at the
true-iTM selector's 65.9% of oracle headroom by construction. Targeting DockQ removes all of it:
the zero-loss optimum is the oracle, 0/415 wrong, 100% of headroom.

AN INTERMEDIATE DESIGN WAS REJECTED and is recorded so it is not re-proposed: quantile-map DockQ
into iTM's range, so the ordering is DockQ's but the values stay "reachable". It gave the same
0/415 and 100%, but it is a permutation of values BETWEEN poses, so two poses with identical DockQ
get different targets depending on where they fell in a global sort -- not even a function of the
pose. An isotonic fit phi(dockq) would fix that, but it was solving the non-problem above.

WHAT THIS COSTS, unchanged by the correction: s_p stops being an error estimate. The model emits
whatever logits make the row-mean transform equal DockQ, so ipTM becomes a selection score and
nothing downstream may read it as a calibrated confidence.

AND WHAT IT FORCES: s_p here is the MEAN over anchor rows, while shipped ipTM and every eval in
this repo select by the MAX. Training the mean and selecting by the max trains a quantity nobody
reads. So the mean-pooling change is not an optional companion experiment -- it has to land with
this, and the existing checkpoints should be rescored under mean pooling FIRST so the two changes
can be told apart (itm_ceiling.py puts the pooling alone at ~7 points of headroom).

NO NEW PARAMETERS AND NO NEW TRANSFORMATION. s_p is pae_row_weight.row_t exactly as it already
exists. A learned head mapping PAE features to DockQ would mean selection no longer reads
chain_pair_iptm, which changes the whole eval path and is ABAG-Rank -- measured at +0.0006
in-sample, -0.0218 held out. Reusing the formula keeps inference unchanged.

ONE BURDEN THIS IMPOSES, stated because it is not obvious: d0 = 1.24*(max(N,19)-15)^(1/3) - 1.8
depends on the interface's token count, and DockQ does not. So the same predicted errors give
different s_p on a large and a small complex, and the model must compensate for d0 to hit the same
DockQ on both. Learnable, but it is work the g_p target did not require.

WHAT EACH PIECE IS FOR, because every one of them answers a measured failure:

  relu, SPLITTING BY DIRECTION. Only two mistakes can change which pose is selected: an acceptable
  pose scored too LOW, and an unacceptable pose scored too HIGH. Scoring a bad pose even lower, or
  a good pose even higher, cannot cost a selection. The relus keep only the two that matter. Note
  the relu comes BEFORE the square -- squaring first would discard the sign and make the split
  meaningless.

  THE HARMLESS DIRECTIONS ARE NEGLIGIBLE, NOT ZERO, and an earlier version of this docstring said
  "contribute no gradient at all", which is false: the relu chooses WHICH WEIGHT multiplies the
  error, not whether there is one. A pose at DockQ 0.80 still carries 1-w = 1.1e-5, so overrating
  it costs ~6.7e-6 against ~2e-1 for underrating it -- a ratio of 3.4e-05. Measured on the real
  data, 0 of 36,900 poses get an exactly-zero coefficient.

  w_p AND (1-w_p), THE ASYMMETRY. They sum to 1, so pose quality slides attention between the two
  directions rather than scaling the loss. A clearly acceptable pose puts all its weight on
  "do not underrate me"; a clearly bad pose puts all of it on "do not overrate me". At DockQ 0.23
  the split is 50/50, which is plain regression for that pose.

  REGRESSION TOWARD g_p RATHER THAN A RANKING TERM. Measured, in docs/dntxmn_results.md: the
  ranking arms lowered their loss by MARGIN COLLAPSE rather than by ordering -- x5frz6 drove
  within-seed ipTM std 0.00593 -> 0.00147 and the top1-top2 gap 0.00205 -> 0.00015 while its loss
  fell. A pairwise loss can always buy relief by refusing to commit, because zero margin is never
  confidently wrong. This loss cannot: g_p is an absolute target, so flattening the scores
  INCREASES it. That is the single most important property here and test_collapse_is_penalised
  pins it.

  SEPARATE NORMALISATION of up and down. If the loss were a mean over all P poses, an interface
  with one acceptable pose out of 50 would give that pose 1/50 = 2% of the gradient, so "call
  everything bad" would be nearly free. Dividing each direction by its own weight mass raises
  that pose's share to about 43% at width 0.05. This is the piece that answers the indifference
  problem, which the PAE loss and plain regression both have.

  THE max(mass, 1) FLOOR ON EACH DENOMINATOR, which the gate forced. Dividing a direction by its
  own mass makes that direction a weighted MEAN, which is scale-free in w -- so a direction whose
  total weight is tiny gets amplified straight back to full strength and the w / (1-w) asymmetry
  does nothing BETWEEN the two directions. Measured by the first run of
  test_asymmetry_is_real: a lone pose at DockQ 0.80 has 1-w = 1.1e-5, but m_down was also 1.1e-5,
  so overrating it cost 0.16000 -- identical to underrating it, the exact thing this loss exists
  to avoid. test_gradient_stops_at_the_target failed for the same reason, with a full-strength
  downward push on a good pose already above its target.

  Flooring each denominator at 1.0 -- one pose's worth of weight -- fixes both without touching
  the case the normalisation was added for: the lone-acceptable-of-50 group has m_up = 2.30, well
  above the floor, so its 43% share is unchanged.

  THE VALUE 1.0 IS NOT PINNED BY THE GATE, and saying it was "forced" overstated it: every floor
  in [0.05, 10] passes all 20 assertions, a 200x range. What the gate forces is that SOME floor
  above ~0.05 exists. 1.0 is a judgement call.

  AND THE FLOOR IS NOT INERT. It switches each term between a weighted-SUM regime (mass < 1) and a
  weighted-MEAN regime (mass > 1), so the per-group loss scale depends on group composition.
  Measured over the 738 real groups at width 0.05: sum(1-w) is below the floor in 262 of them
  (35.5%), where the down term is therefore a plain sum, while sum(w) is below it in only 11
  (1.5%). So the up:down balance is a per-group accident rather than a chosen constant. An
  explicit balance coefficient would at least make it a decision.

  THE MEAN OVER ROWS, NOT THE MAX. Measured by run_logging/itm_ceiling.py on the 738 winnable
  interfaces at 50 candidates: ranking poses by the TRUE iTM recovers 59.0% of the oracle headroom
  with max pooling and 65.9% with mean pooling, mean minus max +0.0107 +- 0.0039 paired (t=2.70).
  So ~7 points of headroom are lost to the pooling alone. The mean also removes the failure
  diagnosed on 9gvn_BF, where the max rested the whole pose's score on one row and, in the
  saturated tail of 1/(1+(c/d0)^2), which row won became arbitrary and the ordering inverted.

KNOWN LIMITATION, stated rather than discovered later. With a 49:1 imbalance the soft weight still
spends 57% of the upward pressure on unacceptable poses at width 0.05, purely by count: each is
only w=0.027, but 49 of them outweigh one acceptable pose. Narrowing to width 0.02 takes the
acceptable pose to 99% but is then effectively a hard threshold at DockQ 0.23. Width 0.05 was
chosen as the compromise and is a dial, not a derived value.

WHAT THIS DOES NOT DO.

FIXED BY TARGETING DockQ, and recorded because the fix is the whole reason for the retarget: with
the true iTM as the target this loss's own global minimum selected an UNACCEPTABLE pose on 40 of
415 mixed groups and strictly preferred that to the correct answer, capping the method at 65.9% of
oracle headroom. With DockQ as the target the zero-loss optimum IS the oracle: 0 of 415 wrong,
100% of headroom. Nothing about the loss's shape changed -- only what it is pointed at.

THE DOWN DIRECTION IS STILL DILUTED, which is the direction that steals the argmax. The up term's
dilution is fixed by the separate normalisation, but a bad pose's coefficient is
(1-w_p)/sum(1-w) ~ 1/P, so ONE bad pose lifting itself above the good one is cheap and gets
cheaper as P grows. At the training P=50 with one acceptable pose: lifting a single bad pose to
0.901 costs 0.0131 against 0.2779 for calling the good pose bad -- 4.7%. The loss does defend
against a wholesale lift (all 49 lifted costs 0.6416), so it is specifically the single-pose steal
that is nearly free, and a single pose is all the argmax needs.

IT DOES NOT DISTINGUISH TWO ACCEPTABLE POSES, by design -- an acceptable pose is unconstrained
from above, so acceptable poses may all saturate near 1. Harmless for selection, but the trained
score is then NOT a calibrated confidence for good poses, and mean DockQ among acceptable poses is
unguided: this loss targets frac acceptable. If mean DockQ matters more, w_p should be dockq_p
itself rather than a sigmoid around the threshold.

SCALE: alpha IS NOT O(1). Measured gradient-norm parity against a PAE cross-entropy on the same
logits needs alpha ~3,000 at the mean real |dL/ds| (2,960) and ~12,200 at the median, because s_p
is a mean over N_rows of a bounded quantity and the squared penalty shrinks the coefficient at the
realistic error scale (median |s-g| = 0.078). The loss VALUE is median 0.0106 against PAE's ~2.0,
so the total-loss column in trajectory.csv will not visibly move -- up, down and both weight
masses must be logged separately or there is no way to tell the term is live. That is the same
class of failure as the row-weight bool cast: a term that logs itself as working while doing
nothing.
"""
import torch

DEFAULT_THRESHOLD = 0.23
DEFAULT_WIDTH = 0.05
_EPS = 1e-8


def quality_weight(dockq, threshold=DEFAULT_THRESHOLD, width=DEFAULT_WIDTH):
    """(w_up, w_down), each in (0, 1) and summing to 1 up to rounding.

    width -> 0 approaches a hard step at `threshold`; large width sends both toward 0.5, which
    makes the loss plain symmetric regression. So the width is a dial between SEPARATION and
    CALIBRATION rather than a choice between them. width must be > 0: at width == 0 a pose exactly
    at the threshold gives 0/0 and NaN propagates silently into the gradient.

    BOTH ARE COMPUTED BY THEIR OWN SIGMOID rather than one as 1 - the other. In float32,
    `1 - sigmoid(x)` underflows to EXACTLY ZERO once sigmoid(x) rounds to 1: at width 0.03 and
    DockQ 0.8 the true value is 5.6e-09 and the subtraction gives 0.0, so the down term vanishes
    identically for every confidently-acceptable pose. That silently removes the "do not overrate"
    pressure at exactly the widths the dilution table recommends (0.02-0.05).
    """
    if not float(width) > 0.0:
        raise ValueError(f"width must be > 0, got {width}")
    z = (dockq - threshold) / width
    return torch.sigmoid(z), torch.sigmoid(-z)


def itm_quality_loss(s, y, dockq=None, w=None, threshold=DEFAULT_THRESHOLD,
                     width=DEFAULT_WIDTH, return_parts=False):
    """Loss for ONE interface's poses. s: [P] with grad; y (= DockQ), dockq: [P] detached.

    `y` IS THE TARGET and is normally the same tensor as `dockq`: the target and the weight come
    from the same quantity, which is what removes the disagreement defect. They are separate
    arguments so an ablation can still regress onto something else (the true iTM, an isotonic
    map) while keeping DockQ weights.

    Pass either `dockq` (weights are derived) or `w` directly. `y` and `w` are detached here
    regardless of what is handed in -- a gradient into the label or the weight would let the model
    lower the loss by moving the target, which is the failure a ranking loss has in a different
    form.

    Each denominator is floored at 1.0, which also removes the inf/nan an all-bad or all-good
    interface would otherwise produce: with no acceptable pose sum(w) is ~0, and such an interface
    legitimately contributes downward pressure only.
    """
    if (dockq is None) == (w is None):
        raise ValueError("pass exactly one of dockq= or w=")
    y = y.detach().to(torch.float32)
    s = s.to(torch.float32)
    if s.shape != y.shape or s.dim() != 1:
        raise ValueError(f"s and y must be the same 1-D shape, got {tuple(s.shape)} "
                         f"and {tuple(y.shape)}")
    if w is None:
        w, w_dn = quality_weight(dockq.detach().to(torch.float32), threshold, width)
    else:
        w = w.detach().to(torch.float32)
        # RANGE IS CHECKED, not assumed. A w outside [0, 1] makes the down term NEGATIVE and the
        # loss can go below zero (w=1.5 gives dn=-0.045), which no amount of training would
        # reveal. The docstring itself suggests passing dockq as w, so hand-built weights are
        # an expected input path.
        if float(w.min()) < 0.0 or float(w.max()) > 1.0:
            raise ValueError(f"w must lie in [0, 1], got [{float(w.min())}, {float(w.max())}]")
        w_dn = 1.0 - w
    if w.shape != s.shape:
        raise ValueError(f"w must match s, got {tuple(w.shape)} and {tuple(s.shape)}")

    # RELU BEFORE THE SQUARE. Each term sees one direction of error and is exactly zero for the
    # other, so the two harmless mistakes contribute no gradient at all rather than a small one.
    under = torch.relu(y - s) ** 2        # acceptable pose scored too low
    over = torch.relu(s - y) ** 2         # unacceptable pose scored too high

    # FLOORED AT ONE POSE'S WORTH OF WEIGHT. See the module docstring: without the floor a
    # direction with tiny total weight is amplified back to full strength and the asymmetry
    # between up and down disappears.
    m_up = w.sum().clamp_min(1.0)
    m_dn = w_dn.sum().clamp_min(1.0)
    up = (w * under).sum() / m_up
    dn = (w_dn * over).sum() / m_dn
    loss = up + dn
    if return_parts:
        # up_share_of_best IS None WHEN THERE IS NO UPWARD PRESSURE. Reporting 0.401 for a group
        # whose up term is exactly 0.0 reads as "40% of the upward pressure sits on one pose"
        # when there is none.
        share = (float(w.max() / w.sum()) if w.numel() and float(w.sum()) > _EPS else None)
        return loss, {"up": up, "down": dn, "w_mass_up": w.sum(), "w_mass_down": w_dn.sum(),
                      "up_share_of_best": share}
    return loss


def dL_ds(s, y, dockq=None, w=None, threshold=DEFAULT_THRESHOLD, width=DEFAULT_WIDTH):
    """dL/ds_p in closed form, for the pass-A path that applies the group loss one pose at a time.

    The existing cross-seed ranking machinery collects per-pose coefficients in a no-grad pass and
    then adds c_p * s_p to each pose's own backward, so only one pose's graph is ever live. This
    loss fits that path because its gradient w.r.t. s_p depends on the other poses ONLY through
    the two weight masses, which pass A can compute without any graph:

        dL/ds_p = -2 w_p relu(y_p - s_p) / max(sum(w), 1)
                  +2 (1-w_p) relu(s_p - y_p) / max(sum(1-w), 1)

    Returned detached, since it IS the coefficient and not part of the graph.
    """
    with torch.no_grad():
        y = y.to(torch.float32)
        s = s.to(torch.float32)
        if w is None:
            w, w_dn = quality_weight(dockq.to(torch.float32), threshold, width)
        else:
            w = w.to(torch.float32)
            w_dn = 1.0 - w
        m_up = w.sum().clamp_min(1.0)
        m_dn = w_dn.sum().clamp_min(1.0)
        return (-2.0 * w * torch.relu(y - s) / m_up
                + 2.0 * w_dn * torch.relu(s - y) / m_dn)
