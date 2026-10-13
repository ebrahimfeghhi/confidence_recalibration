# Review round 4 — structure selection, 2026-10-13

Reviewer's remaining feedback on the cache selection, and their go/no-go.

> I'd fix these two concrete bugs before launching:
> * Select the second representative relative to the actually selected first, avoiding two identical inputs.
> * Normalize antigen-cluster keys so the two-per-pairing cap works correctly.
>
> Once those are fixed and your planned cached-loss replay passes, I'd start the build. The other
> points are refinements; I wouldn't keep delaying the run for them.

Both confirmed on the data and fixed in `2da7b228`, before this round arrived.

## 1. The second representative was ranked against the wrong structure

The loop ranked each candidate against `first_inp`, the first candidate in **sorted** order. That is
not the structure the loop ends up taking, because PDBs already used by another pairing are skipped.
So a pairing could take two structures with identical input while a genuinely different one went
unpicked.

Measured on the selection as it stood: **28 of 69 two-pick pairings held duplicate inputs** —
9b70/9b71, 7tmz/7u9f, 9cz7/9cza and 8hbv/8hbw among them.

Fixed: one pass per pairing, each candidate ranked against what has **actually** been taken, and any
candidate whose `(Hseq, Lseq, sorted polypeptide antigens)` tuple is already present is refused
outright rather than merely deprioritised. **0 duplicate-input pairings remain**; 134 duplicate
candidates were passed over.

## 2. The cap key was not normalised

`agclusters` is `/`-joined **per antigen chain**, with an empty slot for every non-polypeptide
antigen bound to the same complex. One antigen therefore appears as `P220`, `P220/`, `/P220` and
`/P220/` depending on what else is in the structure, and a repeated chain as `P162/P162`. Those were
four distinct cap keys for one pairing, so the two-per-pairing cap did not bind.

Fixed: `ag_key()` splits, drops empties, deduplicates and sorts. **325 raw keys collapse to 243**, and
**0 pairings now exceed the cap**.

## 3. Representatives also now prefer well-resolved interfaces

Taken from the same round's third point, which the reviewer classed as a refinement. Candidates are
ranked by resolution first, then by difference from what is already taken, then by size. The selected
set: median **2.80 Å**, worst 3.50 Å, **62% better than 3.0 Å**. `resolution` and `method` are now
columns in the frozen selection.

## Selection after this round

| | |
|---|---|
| structures | **501** |
| clusters | 193 |
| duplicate-input pairings | 0 |
| pairings over the cap | 0 |
| median resolution | 2.80 Å |
| cache at 10 seeds | 5,010 folds, **1,558 GB** measured |

## Open

Nothing blocking the build except the reviewer's own condition: the **cached-loss replay** must pass
— one cached unit's FT3 loss reproducing the uncached path's loss on the same structure and seed.
Carried from round 3's finding 6, and it is the last gate before 1.56 TB is generated.
