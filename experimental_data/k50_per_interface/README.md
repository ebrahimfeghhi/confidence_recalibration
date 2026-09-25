# k50 -- per-INTERFACE leader selection (SUPERSEDED)

Kept for provenance. Superseded by `top_50_complex`, which selects donors per COMPLEX.

## What this run did

For each interface: order its poses by ipTM, keep a pose only if its pose-pose DockQ to everything
already kept is below 0.23, stop at 50. Self-template each leader (one template per fold, seed
430000).

## Why it is superseded

A template is a whole complex and a fold scores EVERY interface of that complex, so donors cannot
meaningfully be chosen "for interface A-B". Worse, the queue tagged leaders `K00..K49` *per
interface*, so tags collided between siblings of the same complex. Once A-B's `K00` fold had written
rows for every interface under tag `K00`, A-C's `K00` looked already-scored and was skipped.

Net effect: **whichever interface was claimed first supplied the donors for the whole complex**, and
the others silently inherited them. Measured over the 98 interfaces that completed:

| case | interfaces |
|:---|---:|
| single-interface complex -- unaffected | 30 |
| got exactly its own leaders | 25 |
| leaders differ from its own selection | 43 |

Several interfaces got **zero** of their own leaders -- 8rwb_AC, 8w5t_BC, 8w5t_BD, 8yw9_AD, 9bqw_AC.
`8rwb_AC` is the solved interface that dropped 0.718 -> 0.076 in the results table, so that failure
says nothing about the method as designed: it was tested entirely on 8rwb_AB's donors.

## What is still valid

Folds on SINGLE-interface complexes: there per-interface and per-complex selection are identical and
tags cannot collide. `top_50_complex` reuses 428 of them rather than refolding.

The folds themselves are all valid structures. What is untrustworthy is the ATTRIBUTION of a result
to a donor, not the geometry.
