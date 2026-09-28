# Handoff — self-templating vs free sampling

Written 2026-09-25 21:35. Read `README.md` for the pipeline and `AUDIT.md` for every change made on
2026-09-25 and why. This file is the live state.

## The question

Does self-templating on the model's own poses rescue confident decoys better than spending the same
compute on more free seeds? Arms, all selected the same way (top ipTM, DockQ read off the selected pose,
acceptable at DockQ >= 0.23), compared per class by `compare_arms.py`:

| arm | directory | what it is |
|:---|:---|:---|
| free 50 | `experimental_data/free/` | seeds 300000-300049; also the donor pool. **Never rename** (absolute paths everywhere) |
| free + matched | `experimental_data/free_compute_matched/` | extra free seeds, per complex >= the method's donor folds |
| fixed seed | `experimental_data/self_template_fixed_50_seed/` | every donor templated, all on seed 430000 |
| variable seed | `experimental_data/self_template_variable_50_seed/` | the same donors and templates, each on one of the exact free seeds free + matched counts for its complex (`build_varseed_queue.py`) |

Superseded, kept for provenance: `self_template_fixed_50_seed_old_crystal_iptm/` (its still-valid
folds were carried into the new arm) and `self_template_variable_50_seed_old_crystal_iptm/` (never
folded).

## Live state (21:35)

- Selection rebuilt crystal-free: 3,524 donors over 123 complexes. 2,144 folds carried from the old arm
  (25,060 rows, verified identical); **1,380 to fold** (`queue_tofold.tsv`).
- Folding: fixed arm on GPUs 0, 1, 2, 5 (tmux `self_template_fixed_50_seedw<gpu>`). Measured on the
  first 20 folds: 135 s per fold on 8uzp; per-fold cost varies 20 s - 20 min by complex, so no ETA
  from that alone.
- `tmux stagearms` — `diffusion_commit_clean_v2/stage_arms.sh`, per GPU: fixed arm -> the 12-fold free
  top-up (`free_compute_matched/queue_free_topup_arm_a.tsv`) -> variable-seed arm, **started
  automatically** but only once the top-up is fully scored and the variable-seed queue rebuilt from
  scored seeds is identical to the queued one (else the card is held and REMEDIATE logged). Failed
  top-up folds are retried up to 3 times. Variable-seed workers use `--overlap-scoring`.
  `LAUNCH_VARSEED=0` holds every card after the top-up instead. Log `experimental_data/stage_arms.log`.
- `tmux watcharms` — `diffusion_commit_clean_v2/watch_arms.sh`, every 30 min, claims and scored rows
  checked separately. Log `experimental_data/watch_arms.log`.
- Our GPUs are 0, 1, 2, 5. GPU 4 is another user's. **Never more than 4 without asking.** An idle card
  is held with `hold_gpus.py` (tmux `hold<gpu>`); the stager releases it at launch.

## FoldBench, fixed seed (method recorded before launch, 2026-09-26)

Same pipeline, `--dataset foldbench` (see README, Datasets). Inputs `experimental_data/protenix/foldbench/`
from `foldbench_setup.py`: the existing 50-seed Protenix run (seeds 101-150, same model and settings as
pxm2025's pool: protenix-v2, 10 cycles, 200 steps, bf16), every pool pose rescored with our OST scorer
(the run's own table is stale for seeds 101-127). 106 complexes, 163 benchmark interfaces; 8 excluded
with reasons in `setup_audit.md` (8a67, 8dtn, 8evd, 8f60, 8im0, 8sfz, 8ts0; 8sfz is borderline -- its
antigen copies differ only in chromophore, CR2 vs CRO).

- Build: `build_experiment.py --dataset foldbench --out experimental_data/protenix/foldbench/
  self_template_fixed_50_seed --jobs 48`, DOCKQ_GPUS=0,1,2,5. 469 predicted interfaces; donors by the same
  ipTM-order diversity filter, round-robin, cap 50; every donor on seed 430000. No `--diff-against`:
  nothing to carry.
- Fold: `call_folder.py --dataset foldbench --session fb_fixed --overlap-scoring` on GPUs 0,1,2,5, into
  that directory (`poses/`, `results.csv`, `claims/`, `w<gpu>.log`). Scored with `scorer.py --scorer ost`.
- GPUs come from the pxm2025 variable-seed sweep, paused (user's call, 2026-09-26): each worker was
  stopped by PID right after a fold finished, and each log says so. The sweep's queue and claims are kept.
- Resume: tmux `resumevar` waits until no `fb_fixedw` session is left, then relaunches the variable-seed
  sweep on the same four GPUs, the same command it ran with (absolute paths, `--overlap-scoring`).
- Launched 2026-09-27 00:27: 104 complexes, 3,209 folds (`donor_summary.md`), preflight clean. First
  folds verified: a template found on every chain, OST rows in results.csv. Startup burst: the four
  workers used ~255 cores for ~60 s while loading, then ~1 core each -- cap threads on the next launch.
- Order changed 2026-09-27 (user): FoldBench fixed arm -> FoldBench compute-matched free top-up ->
  resume the pxm2025 variable-seed sweep. Top-up: 3,209 donors need 3,209 extra free folds; 1,237 are
  already on disk (seeds outside 101-150 in the original run, plus two later free runs found and verified
  2026-09-27 -- freedeep8pn0, free1k7zau; `foldbench_setup.EXTRA_RUNS` -- all scored into
  `free/results.csv`, same settings: protenix-v2, 10 cycles, bf16); 1,972 new folds over 74 complexes
  (`free_compute_matched/topup_fixed_arm.csv`, `queue_free_topup_fixed.tsv`, session `fb_free`). Other
  archived free-deep runs (`experimental_data/archive/free*_scores.json`) cover only complexes already
  fully matched, and their poses are gone, so they are not used.
- tmux `armchain` runs exactly this (log `experimental_data/resume_variable_sweep.log`):

```bash
# waits for one arm: every queue line claimed and no worker session left
done_() { ! tmux ls 2>/dev/null | grep -q "^$1" && [ $(ls $2/claims 2>/dev/null | wc -l) -ge $(grep -c . $3) ]; }
echo "$(date '+%F %T') chain: FoldBench fixed arm -> FoldBench free top-up (2,072 folds) -> resume the pxm2025 variable-seed sweep" >> /home/ebrahim/tts_abag/experimental_data/resume_variable_sweep.log
until done_ fb_fixedw /home/ebrahim/tts_abag/experimental_data/protenix/foldbench/self_template_fixed_50_seed /home/ebrahim/tts_abag/experimental_data/protenix/foldbench/self_template_fixed_50_seed/queue.tsv; do sleep 60; done
echo "$(date '+%F %T') FoldBench fixed arm finished; launching the FoldBench free top-up" >> /home/ebrahim/tts_abag/experimental_data/resume_variable_sweep.log
/home/ebrahim/Protenix/.venv/bin/python /home/ebrahim/tts_abag/diffusion_commit_clean_v2/call_folder.py --dataset foldbench --queue /home/ebrahim/tts_abag/experimental_data/protenix/foldbench/free_compute_matched/queue_free_topup_fixed.tsv --out /home/ebrahim/tts_abag/experimental_data/protenix/foldbench/free_compute_matched --csv /home/ebrahim/tts_abag/experimental_data/protenix/foldbench/free_compute_matched/results.csv --claim-dir /home/ebrahim/tts_abag/experimental_data/protenix/foldbench/free_compute_matched/claims --log-dir /home/ebrahim/tts_abag/experimental_data/protenix/foldbench/free_compute_matched --session fb_free --gpus 0,1,2,5 --overlap-scoring --run >> /home/ebrahim/tts_abag/experimental_data/resume_variable_sweep.log 2>&1
echo "$(date '+%F %T') launched: $(tmux ls 2>/dev/null | grep -c '^fb_freew') FoldBench free workers" >> /home/ebrahim/tts_abag/experimental_data/resume_variable_sweep.log
until done_ fb_freew /home/ebrahim/tts_abag/experimental_data/protenix/foldbench/free_compute_matched /home/ebrahim/tts_abag/experimental_data/protenix/foldbench/free_compute_matched/queue_free_topup_fixed.tsv; do sleep 60; done
for g in 0 1 2 5; do echo "$(date '+%F %T') === sweep resumed: FoldBench's fixed arm and free top-up finished ===" >> /home/ebrahim/tts_abag/experimental_data/self_template_variable_50_seed/w$g.log; done
/home/ebrahim/Protenix/.venv/bin/python /home/ebrahim/tts_abag/diffusion_commit_clean_v2/call_folder.py --queue /home/ebrahim/tts_abag/experimental_data/self_template_variable_50_seed/queue.tsv --out /home/ebrahim/tts_abag/experimental_data/self_template_variable_50_seed/poses --csv /home/ebrahim/tts_abag/experimental_data/self_template_variable_50_seed/results.csv --claim-dir /home/ebrahim/tts_abag/experimental_data/self_template_variable_50_seed/claims --log-dir /home/ebrahim/tts_abag/experimental_data/self_template_variable_50_seed --templates /home/ebrahim/tts_abag/experimental_data/self_template_fixed_50_seed/templates --gpus 0,1,2,5 --overlap-scoring --run >> /home/ebrahim/tts_abag/experimental_data/resume_variable_sweep.log 2>&1
echo "$(date '+%F %T') relaunched: $(tmux ls 2>/dev/null | grep -c '^self_template_variable_50_seedw') variable-seed workers" >> /home/ebrahim/tts_abag/experimental_data/resume_variable_sweep.log
```

- Then: `compare_arms.py --dataset foldbench` and `analysis/arm_statistics.py --dataset foldbench`.

## When the fixed arm finishes

1. `compare_arms.py --exp experimental_data/self_template_fixed_50_seed` — free 50 vs free + matched vs
   fixed seed. It reports complexes not yet compute-matched and excludes them; after the top-up there
   should be none.
2. The same with `--exp experimental_data/self_template_variable_50_seed` once that arm finishes.
3. Commit the arms' `results.csv` and `index.tsv` at that milestone (not while workers write them).

## Open items

- **Baseline to add: free + matched, uniform** (user, 2026-09-27). The same total number of extra free folds
  as the method spends, spread evenly across complexes instead of per complex by donor count. Separates the
  gain from templating from the gain from where the compute goes: the method gives decoy and nocov
  complexes more folds (median 50 donors) than solved ones (median 24 on FoldBench, 13.5 on pxm2025;
  `donor_summary.md`).

- **The 33 unannotated chains** (22 complexes): the benchmark labels only the chains in its scored
  crystal interfaces, so which chains may form interfaces is still slightly shaped by the crystal.
  Close it by labelling every chain from 1-D metadata, checked by hand, and measuring the change.
- **The anchor:** it only names chains, and the question is whether the top-ipTM anchor does well, not
  whether a worse one would hurt (decided 2026-09-26). Evidence so far: A (top-ipTM anchor) and B
  (crystal naming) give the same donor quality (`analysis/crystal_frame_selection.md`); the arm
  outcome is the real test.
- **Template replacement:** `point_templates_at()` replaces Protenix's 3-4 PDB templates with the one
  self-template rather than adding it, so a rescue could come from the self-template or from removing
  the PDB templates. Separating them needs a third condition.

## Working rules for this user

Lead with the answer, keep it short, one idea at a time when explaining. Report n on every row; per-
interface `chain_pair_iptm`, never global ipTM. Markdown tables. Measure before claiming; say "cannot
measure that yet" rather than estimate. Report, don't interpret. Never launch, kill or exceed 4 GPUs
without approval; kill by PID, never `pkill -f` (it matches its own command line). Never delete poses.
Everything lives in `tts_abag`. Code in `diffusion_commit_clean_v2/`, data in `experimental_data/`.
Docstrings follow SUMMARY / INPUTS / OUTPUTS / EXAMPLE / FUNCTIONS, functions `in: ... out: ...`.
