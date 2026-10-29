"""Write a provenance record beside an analysis script's output, so a rerun can be VERIFIED.

WHY THIS EXISTS. "The script has no RNG and iterates in sorted order" is an argument that it is
reproducible; it is not evidence. These analyses feed numbers into commit messages and into
decisions about what to train next, and the failure mode is silent: an input file is regenerated,
the script is rerun months later, the numbers differ, and nothing says which input moved. This
records enough to tell the difference between "the code changed", "an input changed" and "the
result is genuinely not deterministic".

WHAT IS RECORDED
  - argv, cwd, python version, and the git commit plus whether the tree was dirty
  - for every declared input: size, mtime and a sha256 of the file's bytes
  - for every output: sha256, so two runs can be compared with one diff
  - a digest over the output hashes, which is the single number to compare between runs

HOW TO VERIFY a script is reproducible:

    python -I run_logging/<script>.py ... --out A
    python -I run_logging/<script>.py ... --out B
    python -I run_logging/provenance.py A/provenance.json B/provenance.json

which prints whether the inputs matched and whether the outputs are byte-identical. A differing
output digest with matching input digests is a real non-determinism bug, not an environment
difference.

INPUTS ARE HASHED, NOT JUST PATHED. A path records what was read; a hash records what was in it.
The trunk and label caches in this repo are rebuilt by other scripts, so a path alone would let a
changed input masquerade as a changed result.

DIRECTORIES ARE HASHED OVER THEIR SORTED FILE LIST so a shard added or removed shows up. Large
inputs (the trunk cache is hundreds of GB) are declared by their manifest file, not walked.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

CHUNK = 1 << 20


def sha256_file(path, limit_bytes=None):
    """sha256 of a file's bytes, or of its first limit_bytes when it is huge."""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            if limit_bytes is not None and n + len(b) > limit_bytes:
                h.update(b[: limit_bytes - n])
                break
            h.update(b)
            n += len(b)
    return h.hexdigest()


def _git(root, *args):
    try:
        return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def describe(path, limit_bytes=None):
    """{path, size, mtime, sha256} for a file, or a summary for a directory/glob."""
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
        h = hashlib.sha256()
        tot = 0
        for f in files:
            p = os.path.join(path, f)
            st = os.stat(p)
            tot += st.st_size
            h.update(f.encode())
            h.update(str(st.st_size).encode())
        return {"path": path, "kind": "dir", "n_files": len(files), "bytes": tot,
                "listing_sha256": h.hexdigest()}
    if any(c in path for c in "*?["):
        hits = sorted(glob.glob(path))
        return {"path": path, "kind": "glob", "n_files": len(hits),
                "files": [describe(x, limit_bytes) for x in hits[:64]],
                "truncated": len(hits) > 64}
    if not os.path.exists(path):
        return {"path": path, "kind": "missing"}
    st = os.stat(path)
    return {"path": path, "kind": "file", "size": st.st_size,
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
            "sha256": sha256_file(path, limit_bytes),
            "partial": limit_bytes is not None and st.st_size > limit_bytes}


def write(out_dir, root, inputs, outputs_glob, extra=None, name="provenance.json"):
    """Write the record. inputs: list of paths/globs. outputs_glob: glob of this run's outputs."""
    outs = sorted(glob.glob(os.path.join(out_dir, outputs_glob)))
    o = [{"path": os.path.relpath(p, root), "size": os.path.getsize(p),
          "sha256": sha256_file(p)} for p in outs]
    # THE ONE NUMBER TO COMPARE BETWEEN RUNS: a digest over the output hashes in path order.
    agg = hashlib.sha256()
    for x in o:
        agg.update(os.path.basename(x["path"]).encode())
        agg.update(x["sha256"].encode())
    rec = {
        "script": os.path.basename(sys.argv[0]),
        "argv": sys.argv[1:],
        "cwd": os.getcwd(),
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(root, "status", "--porcelain")),
        "inputs": [describe(p) for p in inputs],
        "outputs": o,
        "output_digest": agg.hexdigest(),
        "determinism": "no RNG; inputs iterated in sorted order; shards are a fixed stride",
    }
    if extra:
        rec.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    with open(p + ".tmp", "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(p + ".tmp", p)
    print(f"  provenance -> {os.path.relpath(p, root)}   output_digest "
          f"{rec['output_digest'][:16]}")
    return rec


def _compare(a_path, b_path):
    A, B = json.load(open(a_path)), json.load(open(b_path))
    print(f"  A {a_path}\n  B {b_path}\n")
    ia = {x.get("path"): x.get("sha256") or x.get("listing_sha256") for x in A["inputs"]}
    ib = {x.get("path"): x.get("sha256") or x.get("listing_sha256") for x in B["inputs"]}
    same_in = ia == ib
    print(f"  inputs identical : {same_in}")
    if not same_in:
        for k in sorted(set(ia) | set(ib)):
            if ia.get(k) != ib.get(k):
                print(f"    DIFFERS {k}\n      A {ia.get(k)}\n      B {ib.get(k)}")
    same_out = A["output_digest"] == B["output_digest"]
    print(f"  outputs identical: {same_out}   ({A['output_digest'][:16]} vs "
          f"{B['output_digest'][:16]})")
    if A.get("git_commit") != B.get("git_commit"):
        print(f"  NOTE different commits: {A.get('git_commit')[:9]} vs "
              f"{B.get('git_commit')[:9]}")
    if same_in and not same_out:
        print("\n  VERDICT: same inputs, DIFFERENT outputs -- a real non-determinism bug.")
        return 1
    if same_in and same_out:
        print("\n  VERDICT: reproducible.")
        return 0
    print("\n  VERDICT: inputs differ, so the outputs are not expected to match.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -I run_logging/provenance.py A/provenance.json "
                 "B/provenance.json")
    sys.exit(_compare(sys.argv[1], sys.argv[2]))
