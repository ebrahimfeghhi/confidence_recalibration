"""Worker for a REAL joint Protenix+reranker CUDA capacity probe.

Launched by torchrun for every (crop size, group size, case) trial. This
provides measurement plumbing, NOT a fake model benchmark: an integration
factory MUST construct the actual trainable Protenix confidence head, original
PAE CE, joint reranker, optimizer, and representative training inputs.

Factory API:
  build_probe(crop_size: int, pose_count: int, case: str, device: torch.device)
      -> (forward_loss_callable, optimizer, head_parameters, reranker_parameters)

forward_loss_callable() returns the complete differentiable
  mean(PAE CE) + alpha*listwise-rank loss for a mixed-label pose group.
The factory owns the actual alpha, temperature, head crop and input setup.
It should initialize DDP if using torchrun with multiple ranks.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import time

import torch


def _factory_from_string(path: str):
    if ":" not in path:
        raise ValueError("--factory must be a Python module:function path")
    module, function = path.rsplit(":", 1)
    fn = getattr(importlib.import_module(module), function)
    if not callable(fn):
        raise TypeError("--factory is not callable")
    return fn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--factory", required=True)
    ap.add_argument("--crop", required=True, type=int)
    ap.add_argument("--poses", required=True, type=int)
    ap.add_argument("--case", default="worst")
    ap.add_argument("--steps", default=3, type=int)
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args()
    if a.poses < 2 or a.steps < 2 or a.crop < 1:
        raise SystemExit("Require poses >= 2, steps >= 2, crop >= 1")

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise SystemExit("GPU probe requires a CUDA device; CPU proxies are invalid")
    dev = torch.device("cuda", local_rank)
    torch.cuda.set_device(dev)
    # Measured BEFORE the model and optimizer are allocated; excludes VRAM
    # occupied by another process. The sweep uses this as the available budget.
    initial_free, total = torch.cuda.mem_get_info(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    metrics_path = Path(f"{a.prefix}.rank{rank}.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "rank": rank, "local_rank": local_rank, "world": world,
        "crop_size": a.crop, "poses": a.poses, "case": a.case,
        "initial_free_bytes": int(initial_free), "total_memory_bytes": int(total),
        "steps": 0, "status": "failed",
    }

    try:
        factory = _factory_from_string(a.factory)
        build = factory(a.crop, a.poses, a.case, dev)
        if not isinstance(build, tuple) or len(build) != 4:
            raise TypeError("factory must return (forward_loss, optimizer, head_params, reranker_params)")
        forward_loss, optimizer, head_params, reranker_params = build
        head_params = tuple(p for p in head_params if p.requires_grad)
        reranker_params = tuple(p for p in reranker_params if p.requires_grad)
        if not head_params or not reranker_params:
            raise ValueError("factory must train BOTH the confidence head and reranker")
        seconds = []
        saw_head_grad = saw_reranker_grad = False
        for i in range(a.steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(dev)
            before = time.perf_counter()
            loss = forward_loss()
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.requires_grad:
                raise ValueError("forward_loss must return a differentiable scalar tensor")
            loss.backward()
            saw_head_grad |= any(p.grad is not None and bool(p.grad.detach().abs().max() > 0)
                                 for p in head_params)
            saw_reranker_grad |= any(p.grad is not None and bool(p.grad.detach().abs().max() > 0)
                                     for p in reranker_params)
            optimizer.step()  # includes AdamW moment/state allocation
            torch.cuda.synchronize(dev)
            seconds.append(time.perf_counter() - before)
            data["steps"] = i + 1
        if not saw_head_grad or not saw_reranker_grad:
            raise RuntimeError("missing live gradients to confidence head or reranker; "
                               "check alpha>0, mixed labels and no-detach")
        data.update({
            "status": "ok",
            "head_grad_live": True,
            "reranker_grad_live": True,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)),
            "max_step_seconds": max(seconds),
            "mean_step_seconds": sum(seconds) / len(seconds),
        })
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        data.update({
            "status": "oom" if ("out of memory" in msg.lower() or "CUBLAS_STATUS_ALLOC_FAILED" in msg)
                           else "error",
            "error": msg,
        })
        raise
    finally:
        # Even when torchrun kills remaining ranks, at least the failing rank
        # leaves a machine-readable record. Never convert failure to success.
        metrics_path.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
