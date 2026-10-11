#!/bin/bash
# Phase 1 search: uniref30_2302 only, index-free. Resumable -- colabfold skips queries whose a3m
# already exists, and the patched build keeps uniref.a3m / prof_res / uniref_tax.m8 so the later env
# pass can resume without uniref30 on disk.
#
# NO INDEX, --db-load-mode 0: measured 0.57 s/query marginal with 314 s fixed overhead, so 5,305
# sequences take ~55 min. A 300 GB index would buy almost nothing at this batch size, and building
# one is what exhausted the quota earlier today.
set -u
R=/home/ebrahim/tts_abag
export COLABFOLD_KEEP_INTERMEDIATES=1
export COLABFOLD_EMIT_TAXID=1
exec taskset -c 0-63 $R/envs/colabfold/bin/colabfold_search \
    --mmseqs $R/tools/mmseqs/bin/mmseqs --threads 64 \
    --use-env 0 --use-templates 0 --db-load-mode 0 \
    $R/extra_training/msa_work/query.fasta \
    $R/tools/colabfold/db \
    $R/extra_training/msa_work/msa_initial
