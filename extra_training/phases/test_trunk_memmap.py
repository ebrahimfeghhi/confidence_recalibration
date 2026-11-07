#!/usr/bin/env python
"""Gate: load_unit(memmap=True) returns BIT-IDENTICAL tensors to load_unit(memmap=False).

WHY A TEST AND NOT AN ARGUMENT. The mapping addresses the trunk npz's bytes directly, at an offset
and a dtype this code works out itself from the zip local header and the npy header. Getting either
wrong does not raise -- it silently hands the confidence head a reinterpretation of the right bytes,
which is the worst failure mode available here. So the two paths are compared element by element on
a REAL trunk, bf16 included (compared through a uint16 view, since bf16 equality would be the thing
under test rather than the test).

CPU ONLY, DELIBERATELY. device="cpu" is the whole point: it runs while the cards are busy, and the
device copy is the only part of load_unit that differs between cpu and cuda. Nothing here needs a
GPU and nothing here allocates one.

    /home/ebrahim/Protenix/.venv/bin/python -I phases/test_trunk_memmap.py
"""
import collections
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_j_rescore_cache as RC                                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUNKS = os.path.join(ROOT, "training/fold/trunks")


def _trunks(n=2):
    """A couple of real trunks, SMALLEST FIRST. The payload is identical in kind at every size and
    a 2 GB trunk would make the test a disk benchmark."""
    paths = sorted(glob.glob(os.path.join(TRUNKS, "*", "*.npz")))
    if not paths:
        return []
    paths.sort(key=lambda p: os.path.getsize(p))
    return paths[:n]


def _same(x, y, name):
    assert type(x) is type(y), f"{name}: {type(x)} vs {type(y)}"
    if torch.is_tensor(x):
        assert x.dtype == y.dtype, f"{name}: dtype {x.dtype} vs {y.dtype}"
        assert x.shape == y.shape, f"{name}: shape {tuple(x.shape)} vs {tuple(y.shape)}"
        # bf16 THROUGH A uint16 VIEW. Comparing bf16 directly would test float equality on the
        # very reinterpretation this gate exists to catch.
        xa = x.view(torch.uint16) if x.dtype == torch.bfloat16 else x
        ya = y.view(torch.uint16) if y.dtype == torch.bfloat16 else y
        assert torch.equal(xa, ya), f"{name}: tensors differ"
        return 1
    if isinstance(x, dict):
        assert set(x) == set(y), f"{name}: keys differ"
        return sum(_same(x[k], y[k], f"{name}.{k}") for k in sorted(x))
    assert x == y, f"{name}: {x!r} vs {y!r}"
    return 1


def main():
    paths = _trunks()
    if not paths:
        print(f"SKIP: no trunk npz under {TRUNKS}")
        return 0
    dev = torch.device("cpu")
    names = ("feat", "s_inputs", "s_trunk", "z_trunk", "poses", "n_cycle")
    for p in paths:
        T, N = collections.defaultdict(float), collections.defaultdict(int)
        plain = RC.load_unit(p, dev)
        mapped = RC.load_unit(p, dev, tick=lambda k, t0: (T.__setitem__(k, T[k] + time.perf_counter() - t0),
                                                          N.__setitem__(k, N[k] + 1)),
                              memmap=True)
        n_cmp = sum(_same(a, b, f"{os.path.basename(p)}:{nm}")
                    for nm, a, b in zip(names, plain, mapped))
        # THE FEATURE MUST CHANGE SOMETHING. A silently-falling-back npy_memmap would pass every
        # equality above while buying nothing, so the mapping itself is asserted to exist.
        mm = RC.npy_memmap(p, "z_trunk")
        assert isinstance(mm, np.memmap), f"{p}: z_trunk did not map"
        assert mm.flags.writeable, "mode='c' must give a writeable array or from_numpy warns"
        assert tuple(mm.shape) == tuple(plain[3].shape), \
            f"{p}: mapped shape {tuple(mm.shape)} vs loaded {tuple(plain[3].shape)}"
        # AND THE TIMERS MUST FIRE, once per member on read and on transfer.
        # 11 members read, 10 transferred: ci_N_cycle is an int the caller reads on the host.
        assert (N["trunk_npz_open"], N["trunk_npz_read"], N["trunk_h2d"]) == (1, 11, 10), \
            f"{p}: tick counts {dict(N)}"
        # THE HOST-DICT PATH, which is what the step-scoped trunk cache hands back and what a
        # worker thread fills. Built ONCE and consumed TWICE, exactly as pass A and pass B do, so
        # a second build from a once-read dict is asserted to equal a fresh read.
        host = RC.read_unit_host(p)
        assert set(host) == set(RC.TRUNK_MEMBERS), set(host) ^ set(RC.TRUNK_MEMBERS)
        from_host_a = RC.load_unit(p, dev, host=host)
        from_host_b = RC.load_unit(p, dev, host=host)
        for nm, x, y in zip(names, plain, from_host_a):
            _same(x, y, f"{os.path.basename(p)}:host:{nm}")
        # WHAT THE NEXT TWO LINES DO *NOT* PROVE. On CPU `.to(cpu)` is a no-op, so both builds
        # ALIAS the host dict and the equality below is trivially true -- it cannot distinguish
        # aliasing from copying, and copying is the property the trunk cache actually depends on.
        # The aliasing is asserted explicitly instead, so this file states what it can and cannot
        # gate: on cuda `.to(cuda)` always copies, and that is NOT tested here because there is no
        # free card. Review of 2026-11-07 caught this assertion being vacuous.
        for nm, x, y in zip(names, from_host_a, from_host_b):
            _same(x, y, f"{os.path.basename(p)}:host-reuse:{nm}")
        assert from_host_a[3].data_ptr() == from_host_b[3].data_ptr(), \
            "on CPU the two builds must alias; if they no longer do, re-read the comment above"
        print(f"OK {os.path.basename(os.path.dirname(p))}/{os.path.basename(p)} "
              f"{os.path.getsize(p)/1e6:.1f} MB: {n_cmp} tensor(s)/scalar(s) bit-identical, "
              f"z_trunk mapped {mm.dtype} {tuple(mm.shape)}, host dict reusable across two builds")
        del plain, mapped, mm, host, from_host_a, from_host_b
    # AND A MEMBER THAT DOES NOT EXIST MUST FALL BACK, not raise.
    assert RC.npy_memmap(paths[0], "no_such_member") is None
    assert RC.npy_memmap("/nonexistent/path.npz", "z_trunk") is None
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
