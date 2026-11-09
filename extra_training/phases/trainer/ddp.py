"""`trainer/ddp.py` -- bring up the 3-rank process group and pick this rank's device.

EXTRACTED VERBATIM from `phases/phase_k_train.py` lines 1713-1772 (source `d34baebc`).
test_bootstrap_equivalence.py compares it line for line.
"""
import time
import os

import torch
import torch.distributed as dist_mod


def setup_dist(a):
    """Returns (rank, world, local_rank, is_main, dist, dev)."""
    # two ways: shards are whole structures truncated to an equal count (shard_by_structure), and
    # each unit is asserted to hold the same number of poses as every other.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = rank == 0
    dist = None
    if world > 1:
        import torch.distributed as dist
        import datetime
        dist.init_process_group(backend=a.dist_backend,
                                timeout=datetime.timedelta(minutes=a.dist_timeout_min))
        torch.cuda.set_device(local_rank)
        print(f"[rank {rank}/{world}] cuda:{local_rank} backend={a.dist_backend} "
              f"timeout={a.dist_timeout_min:.0f} min", flush=True)
    dev = torch.device(f"cuda:{local_rank}") if world > 1 else torch.device("cuda")

    # The effective batch is what the ranks actually sum to, which is not --accum when the world
    # size does not divide it. It is reported rather than silently assumed, and it is what the loss
    # is scaled by, so the gradient is a true mean over the poses that contributed.
    # THE STEP IS K WHOLE STRUCTURES, ONE SEED OF EACH PER RANK. Every seed of a structure crops
    # to the IDENTICAL token count -- the interface crop is defined on the ground-truth interface,
    # not on the predicted pose, and 0 of 488 structures disagree across their ten seeds -- so
    # giving each rank one seed of the same K structures makes the per-step work equal across
    # ranks BY CONSTRUCTION rather than on average. --accum is honoured as a REQUEST: K is the
    # nearest whole number of structures to it.
    structs_per_step = a.structures_per_step or max(
        1, int(round(a.accum / float(world * a.nsample))))
    accum_local = structs_per_step * a.nsample
    accum_eff = accum_local * world
    print(f"  accum: {a.accum} requested -> K={structs_per_step} structure(s)/step x {world} "
          f"rank(s) x {a.nsample} pose(s) = {accum_eff} poses per step ({accum_local}/rank)")
    # CAPTURE WHAT THE CUDA FAILURE ACTUALLY LOOKED LIKE, rather than inferring it afterwards.
    # tv6hue epoch 4 lost two of three held-out shards inside torch._C._cuda_init() and left only
    # the Python traceback, which does not say which card the child had, what its CPU affinity or
    # memory policy were, how much GPU memory was free at that instant, or how long it had been
    # running. Three hypotheses (NUMA inheritance, simultaneous context creation, memory
    # pressure) all survive that traceback, and a probe reproducing the conditions on an idle box
    # initialised 36 times out of 36 -- so the next occurrence has to be the thing that decides
    # it. This costs one line per --loss-only child and nothing in a training run.
    if a.loss_only:
        try:
            # LOCAL IMPORT: the module-level `subprocess` name does not exist -- the only other
            # import of it in this file is inside _git_commit(). Without this the probe would
            # raise NameError, be swallowed by the except below, and print nothing useful, which
            # is the exact failure mode this block exists to prevent.
            import subprocess as _sp
            with open("/proc/self/status") as _fh:
                _st = {l.split(":")[0]: l.split(":", 1)[1].strip() for l in _fh if ":" in l}
            _free = _sp.run(
                ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout
            print(f"  child env: CUDA_VISIBLE_DEVICES="
                  f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}  "
                  f"cpus={_st.get('Cpus_allowed_list')}  mems={_st.get('Mems_allowed_list')}  "
                  f"rss={_st.get('VmRSS')}  torch={torch.__version__}", flush=True)
            print(f"  child sees: {' | '.join(_free.strip().splitlines())}", flush=True)
        except Exception as _e:                   # noqa: BLE001
            print(f"  (child env probe failed: {type(_e).__name__}: {_e})", flush=True)
    _t_cuda = time.time()
    return rank, world, local_rank, is_main, dist, dev
