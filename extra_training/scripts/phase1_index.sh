#!/bin/bash
# Build the uniref30_2302 index. Needed because convertalis only finds taxonomy through
# uniref30_2302_db.idx_mapping / .idx_taxonomy -- index-free search targets _db_seq, which has none,
# so uniref_tax.m8 (and therefore paired MSAs) cannot be produced without it.
set -u
D=/home/ebrahim/tts_abag/tools/colabfold/db
M=/home/ebrahim/tts_abag/tools/mmseqs/bin/mmseqs
cd "$D"
taskset -c 0-63 $M createindex uniref30_2302_db tmp1 --remove-tmp-files 1 --threads 64
echo "EXIT=$?"
ls -la uniref30_2302_db.idx* 
