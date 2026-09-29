# msa_c3 with each donor's own seed (2026-09-29)
User request: same seed (the donor's own, i.e. the seed of the free fold whose trunk is loaded) for the MSA-on, 3-recycle arm, instead of the fixed 430000.
Covers the 6 complexes with saved trunks: batch2 (7yxu, 7zoz, 8be3, 8ux6) and trunk_pilot (9fww, 8fxv), 200 donors. Queues: select/queue_<pdb>_ownseed.tsv (seed field
= donor seed, everything else as queue_<pdb>.tsv). Arm dirs: <batch2|trunk_pilot>/resume/msa_c3_ownseed. PROTENIX_LOG_MC=1 logs the MC-dropout decision per fold.
Control: one 9fww donor at seed 430000 with the dropout log (trunk_pilot/resume/_logmc_control_fixedseed).
Sweep: full stop + relaunch by run_ownseed.sh (tmux os_driver).
