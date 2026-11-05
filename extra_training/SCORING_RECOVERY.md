# Training-fold scoring: what was wrong, what was changed, and how to check it

Scope: the wave-1 training fold, `extra_training/training/fold`. 356 structures x 10 seeds = 3,560
units, 672 pooled interfaces, 5 diffusion samples each = **33,600 expected interface-samples**.

This document exists to be checked, not believed. Every number below has the command that produced
it. The last section lists what is *not* verified.

---

## 1. The headline accounting

| | interfaces | interface-samples |
|---|---|---|
| scored by the pipeline, unaided | 643 (partial) | **32,150** |
| no rows, reference chains genuinely NOT in contact | 12 | 600 |
| no rows, reference chains DO touch -- real gaps | 17 | 850 |
| total | 672 | 33,600 |

`32,150 + 600 + 850 = 33,600`, exactly. Every missing sample is accounted for by one of two
causes, and each missing interface is missing in all 50 of its poses -- there is no partial case.

**97.4% of the scoreable work (32,150 / 33,000) was produced by the pipeline with no intervention
at all**: no rescore pass contributed to it. The only rescore that ran added 0 rows
(`training/runs/train_rescore.log`: `rows 83561 -> 83561 (added 0)`).

Reproduce the table:

```bash
cd extra_training
# the 29 interfaces with no rows, and whether each one's reference chains touch
python -I run_logging/interface_contacts.py --refs training/refs --pairs <the 29>
# expected: "17 of 29 measured interface(s) have at least one heavy-atom contact"
# and the full pool audit, which asserts no SCORED row belongs to a non-contacting pair:
python -I run_logging/pool_contact_audit.py          # exits 1 if any apart_scored exists
```

---

## 2. Defect A -- the pool builder emits pairs that do not touch (600 samples)

`phases/phase_g_build_val_jobs.py` builds the interface pool as a full cross product of antibody
chains against antigen chains:

```python
ab = [ref_of[i] for i, (r, _) in enumerate(chains) if r in ("H", "L")]
ag = [ref_of[i] for i, (r, _) in enumerate(chains) if r.startswith("ag")]
for abc in ab:
    for agc in ag:
        ...
        pool.append(dict(pdb=pdb, chain1=c1, chain2=c2))
```

There is no contact filter. In a structure with two antibodies and two antigen copies, the light
chain of one Fab is paired with the antigen copy it does not bind. `3rhw_DK` is 9.82 A apart;
`4rdq_DF` is 19.25 A apart.

**These were never scoreable and their absence is correct.** They are a defect only because they
inflate the expected count and so made the fold look 4.3% incomplete when it was 2.6% incomplete.

Measured, not assumed. `run_logging/contacts.py` reads `_atom_site` directly and counts heavy-atom
pairs within 5.0 A. The cutoff is calibrated rather than chosen: `6i3z_BC` is scored by OST on 5
contact pairs at 4.27 A, and every unscored pair sits at 6.33 A or beyond, so no threshold between
those two values changes any verdict.

**Status: NOT FIXED.** The filter has to be applied to the existing `pool.csv` after the fold
workers exit, writing every removed pair with its measured distance to a record file. Re-running the
builder from scratch is the wrong move -- it would renumber the pool that 83,561 existing rows refer
to.

---

## 3. Defect B -- our scorer's fallback chain map could never work (100 samples)

`eval/ost_dockq.py` retries a failed OST call with an explicit chain map. The map was built as:

```python
def _identity_chain_map(ref):
    """['A:A','B:B',...] from the reference's chain names, for the -c retry."""
    return [f"{ch.name}:{ch.name}" for ch in st[0]]
```

Every reference chain paired with a model chain **of the same name**. That can only help where the
two files use the same letters, and in this pipeline they never do: reference chains carry the
deposited `label_asym_id`, model chains are named in job order. For `4zto` the reference is D, E, F
and the model is A, B, C.

Worse, `ch.name` is gemmi's *auth* name, so the retry actually sent OST `M:M I:I Q:Q` and OST
refused the call outright rather than scoring badly:

```
RuntimeError: Target chain "M" is not available in target processed for chain mapping ({'D','F','E'})
```

Recorded in `training/fold/poses/4zto/500000-k9sczhnza5/ost_json/0.json`.

Fixing only the namespace would have sent `D:D E:E F:F` and failed just as completely. The premise
was wrong, not the spelling.

### The fix

Match by sequence, which is the one thing a reference chain and its prediction always share:

```python
sm = difflib.SequenceMatcher(None, rs, ms, autojunk=False)
longest = sm.find_longest_match(0, len(rs), 0, len(ms)).size
return longest / min(len(rs), len(ms))
```

One-to-one, and **an ambiguous map produces no map at all**, which skips the retry exactly as a
failed build did. That asymmetry is deliberate: a refusal leaves a visible gap, whereas a forced
wrong map files one interface's DockQ under another interface's name, which nothing downstream can
detect.

Two earlier versions of the metric were wrong, and the second was dangerous:

- `difflib.ratio` divides by the *combined* length. These references are often the variable domain
  alone -- 112 residues against a 217-residue model chain scores 0.68 and is rejected. This alone
  refused 7 of the 8 interfaces used to validate the recovery tool.
- Summing *all* matching blocks lets scattered coincidental matches add up whenever a chain is
  short: `4zto`'s 10-residue peptide antigen `GKAMYAPPIR` scores **1.00 against a 216-residue light
  chain**, tying with the real peptide. That would have built a confidently wrong map rather than
  refusing. Against one contiguous run it is 0.20 for the light chain and 1.00 for the peptide.

Commit `2eb1f2fd`. Deployed into the live scorer while three workers were running: `scorer.py` is a
fresh subprocess per interface, the retry is reached only after a first attempt has already failed,
and the validation fold had written 33,830 OST results with 0 failures, so the path was unreached
there. Compiled before the swap and renamed into place atomically.

### How to check it

```bash
# the map agrees with chain_roles.csv, which is derived from job construction, NOT from sequences
# 12 structures agree, 0 disagree, 2 refused
python -I phases/phase_j_rescore_unmapped.py --targets 4zto:4zto_DF,4zto:4zto_EF \
  --fold training/fold --refs training/refs --roles training/fold/chain_roles_all.csv --seed 500000
```

---

## 4. Defect C -- OST drops an interface when it groups H with L (850 samples)

This one is not our bug. OST sorts reference chains into chemically-equivalent groups, and an
antibody's heavy and light chains are homologous immunoglobulin V domains, so they land in the same
group. When the group has more reference chains than OST has model chains to assign, one reference
chain is left unmapped. It then keeps only interfaces whose *both* chains were mapped:

```python
if i[0] in flat_mapping and i[1] in flat_mapping:
```
`ost/mol/alg/scoring.py:1119`

So the interface never becomes a candidate. Nothing is logged and nothing fails -- the interface is
simply absent from the result, which is why this looked like silence rather than error. For `5tl5`,
OST reports `chem_groups [["A","C"],["B"]]` and `chain_mapping {"A":"B","B":"A"}`, leaving reference
C unmapped.

`-c` cannot override it. OST validates a custom map against its own grouping first:

```
RuntimeError: Chem group mismatch in custom mapping: target chain "D" groups with the following
chemically equivalent target chains: ['B', 'D'] but model chain "B" maps to the following target
chains: ['C']
```

### The fix: pair isolation

Remove the ambiguity before OST sees it. Write a reference cif and a model cif containing **only**
the two chains of the wanted interface. Each chemical group then holds exactly one chain, the
mapping is forced, and DockQ is computed over exactly the pair it is defined over.
`phases/phase_j_rescore_unmapped.py`.

Three guards, because this path produces numbers that nothing else cross-checks:

1. The chain map must be one-to-one above `--min-ident`, or the pose is refused, not guessed.
2. The model chains OST reports for the recovered interface must be the ones `chain_roles.csv`
   says they should be, or the row is rejected. `chain_roles.csv` comes from job construction, so
   this is an independent check, not a restatement of the map.
3. `(iface, seed, sample)` triples already present in `results.csv` are skipped, so the sweep is
   idempotent and cannot create a duplicate. A duplicate would double-weight that pose in every
   average over its seed.

### How to check it -- 16 independent controls

The method's own output cannot validate it, so it was run on interfaces that **already have scores**
and the results compared. 16 controls, chosen across the whole DockQ range, **all 16 reproduced
their stored value exactly**, with ipTM matching to four decimals and the chain attribution correct:

| interface | stored | isolated | | interface | stored | isolated |
|---|---|---|---|---|---|---|
| 31mr_BD | 0.002 | 0.002 | | 9awe_BC | 0.721 | 0.721 |
| 9ixa_BD | 0.015 | 0.015 | | 9tuz_AB | 0.831 | 0.831 |
| 6jhs_CD | 0.045 | 0.045 | | 9mi6_AC | 0.901 | 0.901 |
| 8ib1_AC | 0.142 | 0.142 | | 32nz_BD | 0.459 | 0.459 |

plus 8 more from the first round spanning 0.249 to 0.974.

Two controls were **refused**, correctly: `9b6o` and `9gvn` hold genuinely identical chain copies
that tie at 1.00, and sequence cannot separate those -- only geometry can, which is OST's job.
`3rhw` and `8w2e` refuse for the same reason. That is 4 of 26 structures tried, so symmetric copies
are common enough to matter if this tool is reused.

```bash
# reproduce a control: it must return the value already in results.csv
python -I phases/phase_j_rescore_unmapped.py --targets 9mi6:9mi6_AC \
  --fold training/fold --refs training/refs --roles training/fold/chain_roles_all.csv \
  --seed 500000 --nsample 1
grep '^9mi6_AC,500000,' training/fold/results.csv | head -1
```

---

## 5. Defect D -- our own checker overstated the deficit by 50 samples

`phases/phase_i_check.py verify` reports an interface as `unmapped` before it reaches the contact
measurement, so an interface that is *both* unmapped and non-contacting is reported as a real gap.
`6umg_FH` is 12.67 A apart and was counted as missing work. That is why the deficit was quoted as
900 and is actually 850. The contact check belongs in the unmapped branch too.

**Status: NOT FIXED.**

---

## 6. What is NOT verified

Read this section before trusting the rest.

1. **No recovered row has been written to `training/fold/results.csv`.** Everything above is in
   scratch files. The append is deliberately a separate, approved step.
2. **The 4zto fix is proven on one pose, not 100.** It returns `4zto_DF` 0.906 and `4zto_EF` 0.736
   with the chain assignment `chain_roles` requires. The other 98 poses are produced by the same
   code path but have not been inspected individually.
3. **Pair isolation changes what DockQ is computed over.** With only two chains present, OST has no
   other chains to consider when superposing. On all 16 controls that made no difference to four
   decimal places, which is the strongest evidence available, but it is empirical and not a proof
   that it can never matter.
4. **ipTM for a recovered row is read from the full model's confidence json**, keyed by model chain
   pair -- it is not recomputed on the isolated pair. This is why it matches the stored value
   exactly; it also means ipTM carries no independent information about the isolation.
5. **`chain_roles.csv` is the arbiter of correct attribution.** If it is itself wrong for a
   structure, guard 2 above would confirm a wrong row rather than catch it. It is derived from job
   construction, independently of sequence matching, but it has not been audited separately.
6. **The 5.0 A contact cutoff** is calibrated against OST's own behaviour on one structure
   (`6i3z_BC`, scored at 4.27 A) and a 2 A margin to the nearest unscored pair. It is not OST's
   internal criterion, which was not read.

## 7. Open questions worth a second opinion

- `phases/phase_k_eval.py::ost_chain_pairs()` selects which ipTM to report using the reference
  structure, on `3rhw`, `8w2e`, `9eg6` and `7zra`. ipTM is supposed to be reference-free. This may
  be standard practice for picking which predicted copy to report; I cannot claim that it is.
- `9dc3` has multi-character chain names (`ABID`, `GAID`, `IDVA`) and `run_model/scorer.py` rejects
  any chain pair that is not exactly two characters, so that structure is silently unscoreable.
- Should the 12 non-contacting pairs be deleted from `pool.csv`, or kept with a `no_contact` flag?
  Deleting loses the record that the builder ever proposed them.
