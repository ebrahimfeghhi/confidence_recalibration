# FoldBench self-templating, CROSS-CHAIN MASK ON, 5 recycles

Same donors and templates as `self_template_fixed_50_seed`; each donor on its own compute-matched
free seed. Two things differ from that arm: the mask is ON, and `--ncycle 5` instead of 10.

- `PROTENIX_MULTICHAIN_MASK=1` masks the template's cross-chain features, so a donor's ARRANGEMENT is
  NOT transmitted -- only each chain's internal conformation. `folder.py`'s `TMPL_ENV` pins this to
  `"0"` (pose carried) for the normal arms, so this is the deliberate control.
- 3,209 folds over 104 complexes, ordered decoy complexes first (`--order-by-class`), so the
  interesting results land early and the run can be stopped once they are in.

## Seeds

One distinct seed per donor, from `compare_arms.matched_seeds()` -- free seeds scored on EVERY
interface of that complex -- assigned by `build_varseed_queue.py` with a per-complex deterministic
shuffle. Verified: all 3,209 seeds are scored free folds on every interface of their complex, and none
reuses a first-50 pool seed.

**The seeds are NOT the 151-200 block**, and that is deliberate. `free_compute_matched` (seeds
151-200) covers only 74 of 104 complexes, and ALL 22 complexes with a decoy interface are among the 30
it does not cover -- so restricting to 151-200 would have excluded every interface this experiment is
about. The seeds here come from the wider free pool (0-297) instead; what matters is that each
templated fold has a free twin at the same seed, which the check above confirms.

Beware `build_varseed_queue.py`'s DEFAULT `--extra-csv`: it includes `free/results.csv`, a grab-bag of
697 distinct seeds (0-2036) from many experiments, and `matched_seeds` takes the lowest-numbered. That
is why this arm's seeds start at 0 rather than in a tidy block.

## Fair comparison, deferred

At `--ncycle 5` a templated fold is cheaper than the 10-recycle free folds it is matched against, so a
compute-fair comparison should use roughly half the matched free seeds. Not done here.

## Caveat on attributing any result to the mask

The existing mask-OFF comparator (`self_template_fixed_50_seed`, and the pxm2025 variable arm) ran at
ncycle 10, so mask and recycle count are entangled. "Recycles are epitope search" predicts fewer
recycles would help a wrong-epitope interface and hurt a right-epitope one -- the same direction the
mask is expected to act. The clean control is mask-OFF at ncycle 5, which is a separate run.

Watcher: `diffusion_commit_clean_v2/watch_fb_mask_on.sh`, every 15 min -> `watch.log`.
