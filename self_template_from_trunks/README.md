# self_template_from_trunks

One folder for the whole experiment: self-templating FoldBench antibody-antigen complexes by resuming saved Protenix trunks.

    code/            pipeline.py (the scheduler), patches/ (diffs of every edit made to v2 folder.py and Protenix), history/
    data/<pdb>/      per complex: free/ (50 free folds), select/ (donors, templates), resume/ (the two stage-C arms), summary.json, DONE.json
    kept_trunks/     ONE bf16 z/s trunk per complex (the seed of donor A00), permanent; KEPT.log
    summary_all.csv  one row per interface: free pick vs compute-matched control pick vs templated pick
    controller.log   everything the controller did;  deletions.log  every trunk deletion (path, size)
    archive/         earlier exploration (exploration_pair_rep_scatter = the old pair_rep_scatter folder, symlinked at its old path), test runs, earlier kept trunks

Code it runs: tts_abag/diffusion_commit_clean_v2/ (folder.py edited: 4 env switches) and ~/Protenix (pairformer.py, protenix.py edited). See code/patches/.
Run:  cd code && ~/Protenix/.venv/bin/python pipeline.py run --gpus 0,1,2,5 --cap-gb 450      Status: pipeline.py status      Stop: touch STOP
