# shell_scripts

Watchers and utilities around the phase pipeline. The pipeline itself is in `phases/`.

| file | what it does |
|---|---|
| `watch_train_cache.sh` | watches the training-cache fold, relaunches a dead worker, and hands the GPUs to the FoldBench top-off when the queue drains. Running now as tmux `tcache_watch`. |
| `gpu_guard.sh` | keeps the three cards held so nothing else claims them. Running now. |
| `quota_guard.sh` | stops the fold cleanly if disk use crosses the threshold. Running now. |
| `health.sh` | periodic liveness log for the workers and the box. Running now. |
| `phase2_envdb.sh` | builds the colabfold environmental MSA database, which cannot sit on disk beside uniref30. |
| `lib_fetch.sh` | shared shell helper that downloads a tarball and verifies it against the server's size rather than a constant. |
| `verify_minidb.sh` | proves the index-free MSA search gives identical taxonomy assignments to the indexed one, on a miniature database. |
