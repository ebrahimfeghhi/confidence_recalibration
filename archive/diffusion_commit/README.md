# diffusion_commit

Commit the trunk to a single diffusion pose, then ask whether confidence can finally rank poses.

## The question

A seed's five diffusion poses all come from **one** trunk state. ipTM is a property of that state, so
it cannot rank the five against each other — which is why a confident decoy stays confident. Give each
pose its own trunk state, derived identically, and the confidences become comparable:

1. fold normally for 10 recycles, dump `z_trunk`/`s_trunk`, decode 5 poses (the **donors**)
2. for each donor pose, resume from that same trunk, show the model **that pose as its only template**,
   run **one** further recycle, decode 5 fresh poses

If an arm's ipTM tracks the DockQ of the pose it was shown, that is a selector.

## Arms

| arm | what it is |
|---|---|
| `donor` | the ordinary 10-recycle fold. Its 5 poses are scored *and* become the templates. Donor quality is the x-axis — an earlier version omitted this and the numbers were uninterpretable. |
| `none` | a **stock** 11th recycle: real search templates, default cross-chain mask. Separates "committing to a pose" from "one more recycle does something". |
| `p0..p4` | the same 11th recycle, but the only template is donor pose *k*. |

## Files

| file | |
|---|---|
| `fold_and_commit.sh` | one seed, one interface, all 7 stages. `FOLD_ONLY=1` stops after the fold. |
| `batch.sh` | loops seeds on one GPU, skipping completed ones |
| `build_selftmpl_general.py` | turns a decoded pose into a template job json |
| `row.py` | one CSV row per scored pose |
| `read_results.py` | per-seed summary table |

Code lives here; **data stays** in `confident_decoy_experiments/artifacts/commit11/` — 300 GB+ of it,
not worth moving. Both scripts set `REPO` to point at it.

## Two failure modes that look like success

Both have already cost real GPU time, so the script asserts against them and you should read its output
rather than trust a clean exit.

**A trunk dump is not a successful fold.** `PROTENIX_DUMP_TRUNK` fires inside the forward pass, before
`runner/dumper.py` writes any structure. When the confidence head produced nothing (the
`summary_confidence` KeyError, fixed in Protenix `c5f606a`) the trunk landed and not one CIF did. The
old gate was `[ -s trunk.npz ]`, which passed — so 22 seeds scored arms against zero poses. The fold
now asserts it produced `NS` CIFs and aborts the seed otherwise.

**A silently untemplated arm still produces clean numbers.** The self-pose shares the query's sequence,
so by default it is deduplicated away and the run logs `Found 0 templates` while looking fine. The
p-arms therefore need all of:

- `PROTENIX_TMPL_NODEDUP=1` — stop the self-pose being deduped against the query
- `PROTENIX_MULTICHAIN_MASK=0` + `PROTENIX_TMPL_KEEP_FRAME=1` — keep the inter-chain geometry, which is
  the entire content of a pose; without them the template carries each chain's fold and nothing about
  the dock
- `templatesPath` in the job json — makes the real PDB search hits absent rather than competing

The control deliberately sets none of these; it is meant to be a stock recycle.

## Dropout is on everywhere

i.e. stock. A fixed seed pins the whole RNG stream — two runs at seed 300000 gave identical MC-dropout
decisions and identical MSA draws (6857 rows then 1096, same indices) — so all six arms make the same
decisions as each other. Note this samples **one** point from the dropout distribution rather than
averaging over it; multiple seeds cover that.

## Choosing an interface

The arms need seeds whose 5 donor poses **straddle** DockQ 0.23 — one seed giving both a good and a bad
donor. Measured over the cached FoldBench pool:

| interface | seeds | straddle | good-seed | oracle | ipTM-selected |
|---|---|---|---|---|---|
| **8fas_BC** | 2500 | **44%** | 55% | 0.990 | 0.175 |
| 8che_BE | 50 | 38% | 38% | 0.529 | 0.087 |
| 8bf4_AB | 50 | 4% | 12% | 0.723 | 0.014 |
| 9x1h_EF | 19 | ~0% | 0% | 0.192 | — |

9x1h_EF is the cautionary case: near-zero within-seed spread means the x-axis is flat and no arm can
say anything. Check the straddle rate before spending folds.

## Usage

```bash
# pick target seeds first -- fold only, no arms
GPU=0 PDB=8fas IFACE=8fas_BC SEED=101 FOLD_ONLY=1 bash fold_and_commit.sh

# then the full 7-stage unit on a seed whose donors straddle 0.23
GPU=0 PDB=8fas IFACE=8fas_BC SEED=101 bash fold_and_commit.sh

# or a list of seeds on one GPU
GPU=0 PDB=8fas IFACE=8fas_BC SEEDS='101 102 103' bash batch.sh
```

Target seeds must come from **your own folds**. The same seed re-run gives a different result: seed
300014 of 9x1h_EF spanned DockQ 0.004–0.651 across nine repeats of an identical config. A seed number
from a sweep table does not reproduce.
