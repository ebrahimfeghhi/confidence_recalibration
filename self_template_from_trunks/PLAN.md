# Plan (2026-09-29)

Scope: all 104 FoldBench complexes with benchmark roles, decoy complexes first, then solved, then no-acceptable (classes from the original FoldBench free pool), smallest first within each group; the two decoy giants 7z2m and 7zau close the decoy group (7zau 163 GB of trunks, 8r4u 144, 8xnh 133, 7z2m 129, 8oud 103).
The 6 complexes run earlier (7yxu 7zoz 8be3 8ux6 9fww 8fxv) are rerun in the same pipeline so that everything has the same arms and lives here.

Per complex
  A  50 free folds (seeds 101-150, stock settings, no templates), bf16 z_trunk/s_trunk saved for every fold.
  B  crystal-free donor selection on those new folds (up to 50 donors), templates built. CPU only.
  C  two arms, both resumed from the saved trunks for 3 recycles, MSA on, MC dropout OFF, each fold on its own seed (RNG restored from its trunk):
       T  templated arm: every donor refolded with its pose as the template, from its own seed's trunk.
       K  compute-matched control: same number of folds, same 3 recycles, NO template. Seeds: the donors' distinct seeds first, then further seeds
          from the same 50 (ascending order, never by quality) until the fold count equals the templated arm's.
  Verify, summarize, keep one trunk (seed of donor A00), delete the other trunks. Deletion happens only if rows verified, dropout log complete and none on, summary written, kept trunk copied and matched bit for bit.

Selection everywhere = top-ipTM pose over the arm's folds; acceptable = DockQ >= 0.23. Reported per interface: free run vs K vs T.
Resources: GPUs 0,1,2,5; trunks in flight capped at 450 GB; <= 2 CPU stage-B jobs at 4 cores. Health check + scoring report every 30 min.
Estimated ~246 GPU-h (~2.5-2.8 days on 4 GPUs).
After a complex is finished its per-seed trunks are gone; only the kept trunk remains. Anything needing the other 49 seeds' trunks (other recycle counts, single-row MSA, other gains) must be decided BEFORE launch.
