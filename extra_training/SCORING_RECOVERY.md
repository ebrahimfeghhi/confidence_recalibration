# Training-fold scoring: every defect, every fix, every step

**Branch:** `perf/ckpt-by-size` · **Written:** 2026-11-05 · **For:** independent review

Scope is the wave-1 training fold, [`extra_training/training/fold`](https://github.com/ebrahimfeghhi/confidence_recalibration/tree/perf/ckpt-by-size/extra_training/training).
356 structures x 10 seeds = 3,560 units, 672 pooled interfaces, 5 diffusion samples each =
**33,600 expected interface-samples**.

This document exists to be checked, not believed. Every number has the command that produced it.
[Section 7](#7-what-is-not-verified) lists what is *not* verified, and
[section 6](#6-review-findings-and-what-was-done-about-them) lists what an independent review of my
own code found wrong with it.

---

## 1. The accounting, which closes exactly

| | interfaces | interface-samples |
|---|---|---|
| scored by the pipeline unaided | — | **32,150** |
| no rows, reference chains genuinely NOT in contact | 12 | 600 |
| no rows, reference chains DO touch — real gaps | 17 | 850 |
| **total** | **672** | **33,600** |

`32,150 + 600 + 850 = 33,600`, to the sample. Of the 29 in-scope interfaces with no rows at all,
measurement says 17 touch and 12 do not. Every missing interface is missing in **all 50** of its
poses — there is no partial case anywhere.

**97.4% of the scoreable work (32,150 / 33,000) came out of the pipeline with no intervention of
any kind.** No rescore pass contributed to it: the only one that ran added 0 rows
(`training/runs/train_rescore.log` → `rows 83561 -> 83561 (added 0)`).

Counting against everything the pipeline *set out* to produce, including the 600 pairs the pool
builder should never have proposed, it is 95.7%.

```bash
cd extra_training
python -I run_logging/pool_contact_audit.py     # exits 1 if any SCORED row is a non-contacting pair
python -I run_logging/interface_contacts.py --refs training/refs --pairs <the 29 with no rows>
# => "17 of 29 measured interface(s) have at least one heavy-atom contact"
```

### All 850 are recoverable, by two different routes

| route | interfaces | rows | status |
|---|---|---|---|
| pair isolation ([defect C](#4-defect-c--ost-drops-an-interface-when-it-groups-h-with-l-850-samples)) | 15 | 750 | produced, 50/50 per interface, 0 refusals, 0 mis-attributions |
| full complex through the fixed chain map ([defect B](#3-defect-b--our-scorers-fallback-chain-map-could-never-work-100-samples)) | 2 (`4zto`) | 100 | produced, 100/100 poses, DockQ 0.069–0.965, every one with the model chains `chain_roles` requires |
| | **17** | **850** | **in scratch files, NOT in `results.csv`** |

`4zto_EF` is covered by both routes and they agree exactly, so the production set takes 4zto's 100
from the scorer route and discards pair isolation's duplicate 50, leaving 750 + 100 with no overlap.

---

## How this was worked out, step by step

Chronological, including the two places I was wrong, because those are the parts worth checking
hardest.

1. **Made `verify` able to run at all.** It ignored `--worklist` and enumerated all 856 job files,
   dying on `8gpu` — a structure never queued and so absent from `pool.csv`. Factored the worklist
   reader (`queue_units`) so `verify` and `rescore` share one definition of scope, and gave
   `unmapped` its own bucket instead of folding it into the others. [`64173762`]

2. **Ran the first honest sweep.** 32,150 of 33,600 scored. 1,450 missing, in three buckets:
   800 scored-but-never-mapped, 550 reported as "no contact", 100 never scored at all.

3. **Was challenged on the 550, and had to admit it was an inference.** I had asserted the chains
   don't touch because the scorer found no interface — which is a statement about the scorer, not
   about the structure. The pattern that made it doubtful was that 550 is exactly 11 x 50.

4. **Measured it instead.** Wrote the contact measurement, calibrated the cutoff against a pair OST
   does score (`6i3z_BC`, 4.27 A), and ran it. 0 of the 11 unscored pairs touch, at 6.33–19.25 A;
   33 of 33 scored siblings touch, at 2.21–4.27 A. [`2e49f389`]

5. **Checked the reverse error too**, which is the one that would have corrupted results rather than
   just hidden work: is any *scored* row a pair that does not touch? Audited all 1,711 pool
   interfaces. 0 of 1,711. Then put the measurement inside `verify` so it can never assume this
   again. [`00b21ec4`]

6. **Traced the 550 to the pool builder** — an antibody x antigen cross product with no contact
   filter (defect A).

7. **Diagnosed the 800.** Read OST's own source to find `scoring.py:1119`, confirmed the chem-group
   hypothesis in OST's own error text by trying `-c`, and built pair isolation (defect C).

8. **Validated pair isolation against known answers** — 8 controls, then 10 more on request,
   spanning the full DockQ range. 16 of 16 resolvable controls reproduced exactly; 2 refused as
   genuine identical-copy ties.

9. **Was wrong twice about the sequence metric**, and the second was the dangerous one: summed
   matching blocks scored a 10-residue peptide at 1.00 against a 216-residue light chain, which
   would have built a confidently wrong map rather than refusing. Fixed by requiring a single
   contiguous run.

10. **Diagnosed the last 100 as our own bug**, not OST's — the `name:name` retry map (defect B) —
    fixed it by sequence matching, validated against `chain_roles.csv`, and deployed it into the
    live scorer. [`2eb1f2fd`]

11. **Extended the recovery tool from one pose to all of them**, which is the whole difference
    between a proof and data: 17 rows became 850. [`4f6a0073`]

12. **Ran the full sweep.** 800 of 850 rows, 16 of 17 interfaces complete at 50/50, no refusals and
    no mis-attributions. `4zto_DF` failed all 50 inside OST, and the full-complex route scores it
    instead — which also produced the best cross-check available, since `4zto_EF` is the one
    interface both routes can do and they agree to every digit.

13. **Had my own sweep code reviewed before proposing to write anything**, which found eight
    defects in it (section 6), fixed them, and re-ran the sweep to confirm the output was
    unchanged.

14. **Did not write to `training/fold/results.csv`.** That step is still unauthorised and untaken.

---

## 2. Defect A — the pool builder emits pairs that do not touch (600 samples)

**Where:** [`phases/phase_g_build_val_jobs.py`](https://github.com/ebrahimfeghhi/confidence_recalibration/blob/perf/ckpt-by-size/extra_training/phases/phase_g_build_val_jobs.py) ~line 385

```python
ab = [ref_of[i] for i, (r, _) in enumerate(chains) if r in ("H", "L")]
ag = [ref_of[i] for i, (r, _) in enumerate(chains) if r.startswith("ag")]
for abc in ab:
    for agc in ag:
        c1, c2 = sorted((abc, agc))
        pool.append(dict(pdb=pdb, chain1=c1, chain2=c2))
```

A full cross product of antibody chains against antigen chains, with **no contact filter**. In a
structure with two Fabs and two antigen copies, the light chain of one Fab is paired with the
antigen copy it does not bind. `3rhw_DK` is 9.82 A apart; `4rdq_DF` is 19.25 A apart; `6umg_FH` is
12.67 A apart.

These were never scoreable and **their absence is correct**. They are a defect only because they
inflate the expected count, which made the fold look 4.3% incomplete when it was 2.6% incomplete.

**How it was measured, rather than assumed.** This was the one claim I had asserted without
evidence, and it was challenged. [`run_logging/contacts.py`](https://github.com/ebrahimfeghhi/confidence_recalibration/blob/perf/ckpt-by-size/extra_training/run_logging/contacts.py)
reads `_atom_site` directly and counts heavy-atom pairs within 5.0 A, hydrogens excluded. The
cutoff is calibrated, not chosen: `6i3z_BC` is scored by OST on 5 contact pairs at 4.27 A, and
every unscored pair sits at 6.33 A or beyond, so no threshold between those values changes a single
verdict. Scored siblings read 4 to 414 contact pairs at 2.21–4.27 A; unscored ones read zero.

Chain namespace matters here and cost me an hour: `label_asym_id` is what `pool.csv`, the scorer and
OST all use, while gemmi's `ch.name` is the *auth* name. `3wfc_BC` reads 194 contacts by label and
1,906 by auth, because auth collapses copies. The tool reports both side by side so the
discrepancy cannot hide.

**Status: NOT FIXED.** The filter must be applied to the existing `pool.csv` after the fold workers
exit, writing every removed pair with its measured distance to a record file. Re-running the builder
from scratch is the wrong move — it would renumber the pool that 83,560 existing rows refer to.

---

## 3. Defect B — our scorer's fallback chain map could never work (100 samples)

**Where:** [`eval/ost_dockq.py`](https://github.com/ebrahimfeghhi/confidence_recalibration/blob/perf/ckpt-by-size/eval/ost_dockq.py) · **Fixed in** [`2eb1f2fd`](https://github.com/ebrahimfeghhi/confidence_recalibration/commit/2eb1f2fd)

When OST's automatic chain mapping fails, the scorer retries with an explicit map. The map was:

```python
def _identity_chain_map(ref):
    """['A:A','B:B',...] from the reference's chain names, for the -c retry."""
    return [f"{ch.name}:{ch.name}" for ch in st[0]]
```

Every reference chain paired with a model chain **of the same name**. That can only help where both
files use the same letters, and in this pipeline they never do: reference chains carry the deposited
`label_asym_id`, model chains are named in job order. For `4zto` the reference is D, E, F and the
model is A, B, C.

Worse, `ch.name` is gemmi's *auth* name, so the retry actually sent OST `M:M I:I Q:Q` and OST
refused the call outright rather than scoring badly:

```
RuntimeError: Target chain "M" is not available in target processed for chain mapping ({'D','F','E'})
```

Recorded in `training/fold/poses/4zto/500000-k9sczhnza5/ost_json/0.json`. Fixing only the namespace
would have sent `D:D E:E F:F` and failed just as completely — the premise was wrong, not the
spelling.

### The fix: match by sequence

The one thing a reference chain and its prediction always share is the amino acid sequence.

```python
sm = difflib.SequenceMatcher(None, rs, ms, autojunk=False)
longest = sm.find_longest_match(0, len(rs), 0, len(ms)).size
return longest / min(len(rs), len(ms))
```

One-to-one, and **an ambiguous map produces no map at all**, which skips the retry exactly as a
failed build did. That asymmetry is deliberate: a refusal leaves a visible gap, whereas a forced
wrong map files one interface's DockQ under another interface's name, and nothing downstream can
detect that.

**Two earlier versions of the metric were wrong, and the second was dangerous.**

- `difflib.ratio` divides by the *combined* length. These references are often the variable domain
  alone — 112 residues against a 217-residue model chain scores 0.68 and is rejected. This alone
  refused 7 of the 8 interfaces used to validate the recovery tool.
- Summing *all* matching blocks lets scattered coincidental matches add up whenever a chain is
  short. `4zto`'s 10-residue peptide antigen `GKAMYAPPIR` scores **1.00 against a 216-residue light
  chain**, tying with the real peptide. That would have produced a confidently wrong map instead of
  a refusal. Against one contiguous run it is 0.20 for the light chain and 1.00 for the peptide.

**Deployed into the live scorer while three workers were running**, deliberately: `scorer.py` is a
fresh subprocess per interface, the retry is reached only after a first attempt has already failed,
and the validation fold had written 33,830 OST results with 0 failures, so the path was unreached
there. Compiled before the swap and renamed into place atomically.

**Validated two ways.** The map agrees with
[`chain_roles.csv`](https://github.com/ebrahimfeghhi/confidence_recalibration/tree/perf/ckpt-by-size/extra_training/training)
on 12 structures — 12 agree, 0 disagree, 2 correct refusals — and `chain_roles.csv` is derived from
job construction, not from sequences, so it is an independent check. End to end on `4zto` the old
scorer returns nothing and this returns `4zto_DF` 0.906 and `4zto_EF` 0.736 on model chains `(B,C)`
and `(A,C)` as the roles require.

---

## 4. Defect C — OST drops an interface when it groups H with L (850 samples)

This one is not our bug.

OST sorts reference chains into chemically-equivalent groups, and an antibody's heavy and light
chains are homologous immunoglobulin V domains, so they land in the same group. When the group holds
more reference chains than OST has model chains to assign, one reference chain is left unmapped. It
then keeps only interfaces whose *both* chains were mapped:

```python
if i[0] in flat_mapping and i[1] in flat_mapping:
```
`ost/mol/alg/scoring.py:1119` (OpenStructure 2.8.0)

So the interface never becomes a candidate. Nothing is logged and nothing fails — it is simply
absent from the result, which is why this presented as silence rather than error. For `5tl5`, OST
reports `chem_groups [["A","C"],["B"]]` and `chain_mapping {"A":"B","B":"A"}`, leaving reference C
unmapped.

**`-c` cannot override it.** OST validates a custom map against its own grouping first:

```
RuntimeError: Chem group mismatch in custom mapping: target chain "D" groups with the following
chemically equivalent target chains: ['B', 'D'] but model chain "B" maps to the following target
chains: ['C']
```

### The fix: pair isolation

**Where:** [`phases/phase_j_rescore_unmapped.py`](https://github.com/ebrahimfeghhi/confidence_recalibration/blob/perf/ckpt-by-size/extra_training/phases/phase_j_rescore_unmapped.py)

Remove the ambiguity before OST sees it. Write a reference cif and a model cif containing **only**
the two chains of the wanted interface. Each chemical group then holds exactly one chain, the
mapping is forced, and DockQ is computed over exactly the pair it is defined over.

Three guards, because this path produces numbers nothing else cross-checks:

1. The chain map must be one-to-one above `--min-ident`, or the pose is refused, not guessed.
2. The model chains OST reports must be the ones `chain_roles.csv` says, or the row is rejected.
3. `(iface, seed, sample)` triples already in `results.csv` are skipped, so the sweep is idempotent.
   A duplicate would double-weight that pose in every average over its seed.

### Validated on 16 interfaces that already had scores

The method's own output cannot validate it, so it was run on interfaces that **already have rows**
and the results compared. 16 controls across the whole DockQ range, **all 16 reproduced their
stored value exactly**, ipTM to four decimals, chain attribution correct:

| interface | stored | isolated | | interface | stored | isolated |
|---|---|---|---|---|---|---|
| 31mr_BD | 0.002 | 0.002 | | 9awe_BC | 0.721 | 0.721 |
| 9ixa_BD | 0.015 | 0.015 | | 9tuz_AB | 0.831 | 0.831 |
| 6jhs_CD | 0.045 | 0.045 | | 9mi6_AC | 0.901 | 0.901 |
| 8ib1_AC | 0.142 | 0.142 | | 32nz_BD | 0.459 | 0.459 |

plus 8 more from the first round spanning 0.249 to 0.974.

**Two controls were refused, correctly.** `9b6o` and `9gvn` hold genuinely identical chain copies
that tie at 1.00; sequence cannot separate those, only geometry can, which is OST's job. `3rhw` and
`8w2e` refuse for the same reason. That is 4 of 26 structures tried, so symmetric copies are common
enough to matter if this tool is reused.

```bash
# reproduce a control: it must return the value already in results.csv
python -I phases/phase_j_rescore_unmapped.py --targets 9mi6:9mi6_AC \
  --fold training/fold --refs training/refs --roles training/fold/chain_roles_all.csv --seed 500000
grep '^9mi6_AC,500000,' training/fold/results.csv | head -1
```

---

## 5. Defect D — my own checker overstated the deficit by 50 samples

**Where:** [`phases/phase_i_check.py`](https://github.com/ebrahimfeghhi/confidence_recalibration/blob/perf/ckpt-by-size/extra_training/phases/phase_i_check.py)

`verify` reports an interface as `unmapped` before it reaches the contact measurement, so an
interface that is *both* unmapped and non-contacting is reported as real missing work. `6umg_FH` is
12.67 A apart and was counted as a gap. That is why the deficit was quoted as 900 and is actually
850. The contact check belongs in the unmapped branch too.

**Status: NOT FIXED.**

---

## 6. Review findings, and what was done about them

I had a review run against my own sweep code before proposing to write anything. It found eight
defects. None affected the 800 rows already produced — verified: all 800 have `iface` matching their
`pdb`, and the skip guard did load (83,550 keys) — but several would have mattered on the next use.
All are fixed in the commit that carries this document.

| # | finding | severity | fix |
|---|---|---|---|
| 1 | `--targets 9vb4:9vaw_AB` would file 9vb4's DockQ under 9vaw's name; every guard passes because they all consult 9vb4 | **wrong number, undetectable** | `parse_targets` refuses unless the iface starts with `<pdb>_` |
| 1b | `<pdb>_DA` names the same interface as `<pdb>_AD`, matches neither its rows nor the skip guard | duplicate under a second name | refuses chains not in sorted order, as `pool.csv` writes them |
| 2 | a non-existent `--skip-existing` path returned an empty set, silently disabling the whole duplicate guard | **doubles every row on append** | hard error; the guard protecting the data must not degrade quietly |
| 3 | a zero-row run left the previous run's `--rows-out` in place, so a sweep-then-append caller appends it twice | duplicate rows | always written, header-only when empty |
| 4 | the shared reference subset had no per-invocation identity; concurrent shards would truncate it mid-read | **silently low DockQ** | pid in the name, and `write_subset` now writes to a temp and `os.replace`s |
| 5 | `skipped` counted nothing when a pose was partly done | misleading summary | `skipped += len(ifaces) - len(todo)`, unconditionally |
| 6 | exit 1 on any refusal, so a fully successful sweep failed and `sweep && append` aborted on good rows | **blocks correct output** | 1 only for real failures; 2 for a sweep that did no work; refusals and unfolded seeds are reported, not failed |
| 7 | a missing reference `continue`d with no record — "0 failure(s)", exit 0 | **work vanishes silently** | recorded in `failed` |
| 8 | `--seeds 10` without `--seed0` swept one seed and looked like a clean run | **silent under-run** | hard error |

Lower severity, also fixed: a stale OST json at the same path could be accepted as a fresh result
(`invoke` has no skip-if-exists and OST can exit 0 writing nothing), so the output is unlinked
first; a blank `iptm` is now reported; duplicate `--targets` tokens are de-duplicated; and two pose
dirs for one seed is a refusal rather than an arbitrary pick.

**The fixes changed no numbers.** The full sweep was re-run after them and its 800 canonical rows
are byte-identical to the pre-fix run (`diff` clean), with the new exit code correctly returning 1
for the 50 genuine `4zto_DF` OST failures and no longer for refusals.

Every guard was then checked by actually invoking it:

```
9vb4:9vaw_AB       -> interface 9vaw_AB does not belong to 9vb4
4zto:4zto_FD       -> chains must be in sorted order (4zto_DF)
bad skip path      -> no such file. Refusing to sweep without the duplicate guard
--seeds w/o seed0  -> --seeds > 1 needs --seed0
```

### A pre-existing problem the review surfaced

`training/fold/results.csv` holds **83,560 rows but only 83,550 distinct `(iface, seed, sample)`
keys** — 10 duplicated poses:

```
7t86_EI/500009/0  7uij_DF/500007/3  8ag1_BE/500004/4  8ulj_BD/500002/4  8v91_CF/500006/1
8w2e_BC/500007/1  9ejz_AC/500006/4  9hkq_BC/500004/1  9hv9_AC/500007/0  9iut_DF/500004/3
```

These are already known to [`run_logging/dedupe_results.py`](https://github.com/ebrahimfeghhi/confidence_recalibration/blob/perf/ckpt-by-size/extra_training/run_logging/dedupe_results.py),
whose docstring names them and which resolves all 10 against the pose still on disk rather than by
row order. But nothing has ever removed them, so those 10 poses still double-weight in any average
over their seed. Unrelated to this recovery; outstanding on its own.

---

## 7. What is NOT verified

Read this before trusting the rest.

1. **No recovered row has been written to `training/fold/results.csv`.** Everything here is in
   scratch files. The append is deliberately a separate step and has not been taken.
2. **`4zto_DF` cannot be pair-isolated at all.** All 50 poses fail inside OST with
   `IndexError: arrays used as indices must be of integer (or boolean) type` on a two-chain input
   where a 10-residue peptide faces a 113-residue domain. Its sibling `4zto_EF` isolates fine with
   the same peptide, and **I cannot explain why one and not the other.** It does not block the
   recovery, because 4zto is a defect-B case that the full-complex path scores, but the asymmetry
   is unexplained.
3. **Pair isolation changes what DockQ is computed over.** With two chains present, OST has no
   others to consider when superposing. 16 controls say that makes no difference to four decimals,
   and `4zto_EF` — the one interface both routes can do — returns identical values by each route
   (0.736, 0.747, 0.747 and ipTM 0.5637, 0.5692, 0.5653). That is strong, but empirical.
4. **ipTM on a recovered row is read from the full model's confidence json**, keyed by model chain
   pair; it is not recomputed on the isolated pair. That is why it matches exactly, and it also
   means ipTM carries no independent information about the isolation.
5. **`chain_roles.csv` is the arbiter of correct attribution.** If it is wrong for a structure,
   guard 2 would confirm a wrong row rather than catch it. It is derived independently of sequence
   matching, but it has not been audited on its own.
6. **The 5.0 A contact cutoff** is calibrated against OST's behaviour on one structure and a ~2 A
   margin to the nearest unscored pair. It is not OST's internal criterion, which I did not read.
7. **4zto's 100 rows have not been produced by the pipeline.** They were produced by a scratch
   diagnostic. The right producer is `phase_i_check rescore`, which runs the real `scorer.py`; since
   the fix is deployed and the poses are cached it should now succeed where it wrote FAILURE records.

---

## 8. Open questions worth a second opinion

- [`phases/phase_k_eval.py`](https://github.com/ebrahimfeghhi/confidence_recalibration/blob/perf/ckpt-by-size/extra_training/phases/phase_k_eval.py)`::ost_chain_pairs()`
  selects which ipTM to report using the **reference** structure, on `3rhw`, `8w2e`, `9eg6` and
  `7zra`. ipTM is supposed to be reference-free. This may be standard practice for choosing which
  predicted copy to report; I cannot claim that it is.
- `9dc3` has multi-character chain names (`ABID`, `GAID`, `IDVA`) and
  [`run_model/scorer.py`](https://github.com/ebrahimfeghhi/confidence_recalibration/blob/perf/ckpt-by-size/run_model/scorer.py)
  rejects any chain pair that is not exactly two characters, so that structure is silently
  unscoreable.
- Should the 12 non-contacting pairs be deleted from `pool.csv` or kept with a `no_contact` flag?
  Deleting loses the record that the builder ever proposed them.

---

## 9. Commits, in order

| commit | what it did |
|---|---|
| [`b6affe12`](https://github.com/ebrahimfeghhi/confidence_recalibration/commit/b6affe12) | measured forced chain mapping against OST's automatic one on the same poses — the first evidence the mapping was the problem |
| [`64173762`](https://github.com/ebrahimfeghhi/confidence_recalibration/commit/64173762) | `verify`: one scope reader for both modes, and the `unmapped` bucket counted on its own |
| [`2e49f389`](https://github.com/ebrahimfeghhi/confidence_recalibration/commit/2e49f389) | measured whether the 550 "no contact" samples really have no contact |
| [`00b21ec4`](https://github.com/ebrahimfeghhi/confidence_recalibration/commit/00b21ec4) | shared `contacts.py`; `verify` checks the claim instead of assuming it; audits every pool pair for the reverse error |
| [`2eb1f2fd`](https://github.com/ebrahimfeghhi/confidence_recalibration/commit/2eb1f2fd) | the scorer's `-c` retry builds its map by sequence, not by assuming the names match |
| [`4f6a0073`](https://github.com/ebrahimfeghhi/confidence_recalibration/commit/4f6a0073) | recover every pose of a dropped interface, not one; first version of this document |

## 10. Files written or changed

| file | what it is |
|---|---|
| [`run_logging/contacts.py`](https://github.com/ebrahimfeghhi/confidence_recalibration/blob/perf/ckpt-by-size/extra_training/run_logging/contacts.py) | **new.** The shared measurement: heavy-atom pairs within a cutoff, read from `_atom_site`, in either chain namespace |
| [`run_logging/interface_contacts.py`](https://github.com/ebrahimfeghhi/confidence_recalibration/blob/perf/ckpt-by-size/extra_training/run_logging/interface_contacts.py) | **new.** CLI that reports contacts per interface in both namespaces, `--control` adds every sibling pool interface |
| [`run_logging/pool_contact_audit.py`](https://github.com/ebrahimfeghhi/confidence_recalibration/blob/perf/ckpt-by-size/extra_training/run_logging/pool_contact_audit.py) | **new.** Crosses measured contact against rows present over all 1,711 pool interfaces; exits 1 if any non-contacting pair was ever scored |
| [`phases/phase_j_rescore_unmapped.py`](https://github.com/ebrahimfeghhi/confidence_recalibration/blob/perf/ckpt-by-size/extra_training/phases/phase_j_rescore_unmapped.py) | the recovery tool: pair isolation, sequence-matched map, role cross-check, full seed/sample sweep, canonical row output |
| [`eval/ost_dockq.py`](https://github.com/ebrahimfeghhi/confidence_recalibration/blob/perf/ckpt-by-size/eval/ost_dockq.py) | the `-c` retry map, rebuilt by sequence. **Deployed live** |
| [`phases/phase_i_check.py`](https://github.com/ebrahimfeghhi/confidence_recalibration/blob/perf/ckpt-by-size/extra_training/phases/phase_i_check.py) | `verify` honours `--worklist`, buckets `unmapped` separately, and measures contact instead of assuming it |
