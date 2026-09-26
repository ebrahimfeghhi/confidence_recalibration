"""Run provenance: a stable id, a run-scoped output directory, and a run.json per fold.

WHY THIS EXISTS. Folds are not reproducible run-to-run (bf16 reductions reorder; the same seed has
landed at DockQ 0.03 and 0.59 on the same card), so a repeat fold of a seed is DATA, not a duplicate
-- it must not overwrite the first. Every fold therefore gets its own directory named by a random id,
plus a run.json recording enough to tell two runs of the same seed apart after the fact: precision,
recycle count, which poses were used as templates and their checksums.

The pxm2025 pool was written without any of this, and its exact configuration is now unrecoverable --
we cannot tell from disk whether it was bf16 or fp32, or what templates it saw. That is the failure
this module is meant to prevent repeating.
"""
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import time


ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
_RNG = random.SystemRandom()   # immune to random.seed(); see run_id()


def run_id(defining: dict = None, n: int = 10) -> str:
    """A random 10-char id naming ONE fold's directory.

    RANDOM, not a hash of the inputs. Two folds of the same seed are different data -- folds do not
    reproduce -- so an id derived from the inputs would collapse them onto one directory, and the
    second would overwrite the first. 36^10 is large enough that collisions are not a consideration;
    the previous scheme hashed into 36^3 for TEMPLATE ids and 720 of ~3300 collided.

    OS entropy, NOT the global `random` module. Protenix calls random.seed(<fold seed>) before every
    fold, so an id drawn from the global generator depends only on the previous fold's seed. Every
    fold of the self-template arms uses seed 430000, and 173 of its folds landed in 27 directories,
    each overwriting the last one's poses and run.json (found 2026-09-25).

    `defining` is accepted and ignored so old callers keep working.
    """
    return "".join(_RNG.choice(ALPHABET) for _ in range(n))


def run_dir(out: str, pdb: str, seed, rid: str) -> str:
    """<out>/<pdb>/<seed>-<run_id>/ -- one directory per fold.

    The seed is in the NAME so `ls` is useful (which seeds are done, which are missing), and in
    run.json where it is authoritative -- a renamed directory must not be able to lie about it.

    The experiment name is the caller's --out, so the old <mode> level is gone: "periodic_2_5_8" as
    an experiment name says more than "free" did as a mode. Its absence also fixed a real accident --
    passing --out <tree>/free while the mode was also "free" produced <tree>/free/free/<pdb>/...

    NO INTERFACE LEVEL. Every fold produces the WHOLE complex, whatever it was aimed at, so an
    interface in the path is misleading: it suggests the output is specific to one interface when the
    coordinates cover all of them. Which interface motivated a templated arm, and which poses were
    used, live in run.json instead -- metadata, not directory structure. The earlier shape also
    invited a silent error: the old driver passed pdb into the interface slot, so pool folds landed
    at <pdb>/<pdb>/ while the new one wrote <pdb>/all/, two shapes for one kind of fold.""" 
    return os.path.join(out, str(pdb), f"{seed}-{rid}")


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_run_json(rdir: str, **fields) -> str:
    """Record everything needed to tell this fold apart from another run of the same seed.

    Callers pass what they know (mode, pdb, iface, seed, arm, dtype, n_cycle, n_sample, templates,
    namespace, driver_args); the environment facts that turn out to matter -- host, visible devices,
    layernorm kernel, git sha -- are filled in here so no caller can forget them.
    """
    os.makedirs(rdir, exist_ok=True)
    doc = {
        "run_id": os.path.basename(rdir.rstrip("/")),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "layernorm_type": os.environ.get("LAYERNORM_TYPE"),
        "git_sha": git_sha(),
        "argv": " ".join(sys.argv[1:]),
        **fields,
    }
    p = os.path.join(rdir, "run.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, p)          # atomic: a reader never sees a half-written run.json
    return p
