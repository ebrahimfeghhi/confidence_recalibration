# The fold worker moved to `run_model/`

2026-10-14. These four files left this directory:

| file | now at |
|---|---|
| `folder.py` | `run_model/folder.py` |
| `scorer.py` | `run_model/scorer.py` |
| `call_folder.py` | `run_model/call_folder.py` |
| `reconcile_stale_claims.py` | `run_model/reconcile_stale_claims.py` |

They moved together because each finds the next by LOCATION, not by import name: `folder.py` spawns
`{dirname(__file__)}/scorer.py`, `call_folder.py` does `cd {its own dir} && python folder.py`, and
`reconcile_stale_claims.py` adds its own directory to `sys.path` before `import folder`. Moving any
one of them alone breaks it.

**Older launchers in this repo still say `cd diffusion_commit_clean_v2 && python folder.py` and will
fail.** That was accepted deliberately: they are records of runs that already finished, and their
results are in the CSVs regardless. Repoint one if you need to re-run it.

What stayed here and is still used: `build_free_pool.py`, `build_topup_queue.py`, the block watchers
(`watch_*.sh`), the queue builders and `analysis/`.
