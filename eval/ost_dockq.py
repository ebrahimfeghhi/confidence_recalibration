#!/usr/bin/env python
"""THE OST DockQ scorer. Every OST score in this project goes through here.

WHY ONE FILE. `run_ost` was copy-pasted into 14 scripts and the copies drifted. Five of them still
passed `--rigid-scores`, which crashes on interfaces with fewer than three superposable points
("3 points required for superposition"). The crash kills the whole call before the json is written,
so a DockQ that had computed fine was never recorded and the caller logged a 0. That is how 8c5h's
real DockQ of 0.899 ended up in the failure list.

`--rigid-scores` IS SAFE TO DROP -- verified, not assumed. Running the same model/reference pair both
ways gives byte-identical `dockq`, `dockq_interfaces`, `fnat`, `irmsd`, `lrmsd`, `dockq_ave` and
`chain_mapping`. The flag adds exactly four keys -- `oligo_gdtha`, `oligo_gdtts`, `rmsd`, `transform`
-- and nothing in this project reads any of them. So the flag bought nothing and cost whole scores.

CHAIN LETTERS. `interfaces` is keyed on REFERENCE chain names, and `model_chains` gives the mapping to
model chain names for that interface. They are not interchangeable: confusing the two once put 45% of
this project's interface measurements on the wrong chains. ipTM is indexed by MODEL asym order, which
is why `pair_iptm` needs the mapping rather than the reference letters.

Usage as a library:
    from ost_dockq import score
    r = score(model_cif, ref_cif, out_json)          # -> {"A,C": {"dockq":..., "model_chains":...}}
    r = score(model_cif, ref_cif, out_json, conf_json=cj)   # adds "pair_iptm"
    v = interface(r, "A", "C")                        # that interface, in either order or copy

Usage from the shell:
    ost_dockq.py --model m.cif --ref r.cif --out o.json [--conf summary_confidence.json]
"""
import argparse
import json
import os
import subprocess

OST = os.environ.get("OST_BIN", "/home/ebrahim/miniconda3/envs/foldbench/bin/ost")
# OST spawns threads per structure; unpinned, several concurrent jobs oversubscribe the box and each
# runs slower than if they had been serialised.
TASKSET = os.environ.get("OST_TASKSET", "0-39")
TIMEOUT = int(os.environ.get("OST_TIMEOUT", "300"))
# OST's binary needs its conda env's libs on the path when invoked outside that env.
OST_LIB = os.environ.get("OST_LIB", "/home/ebrahim/miniconda3/envs/foldbench/lib")
ENV = {**os.environ, "LD_LIBRARY_PATH": OST_LIB}


def invoke(model, ref, out, extra=None, chain_map=None, timeout=TIMEOUT):
    """ONE OST call. No skip-if-exists, no retry -- use this when you need to control both.

    Success is `returncode == 0` AND a non-empty json: OST can exit 0 having written nothing at all,
    and treating that as success is how empty results got recorded as scores of zero.
    """
    cmd = []
    if TASKSET:
        cmd += ["taskset", "-c", TASKSET]
    cmd += [OST, "compare-structures", "-m", str(model), "-r", str(ref), "-o", str(out),
            "--fault-tolerant", "--min-pep-length", "4", "--min-nuc-length", "4"]
    cmd += [f for f in (extra or []) if f not in ("--rigid-scores", "--tm-score")] or ["--dockq"]
    if chain_map:
        cmd += ["-c", *chain_map]
    os.makedirs(os.path.dirname(str(out)) or ".", exist_ok=True)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, env=ENV)
    except Exception:        # TimeoutExpired / OSError -- let the caller decide about a fallback
        return False
    return r.returncode == 0 and os.path.exists(out) and usable(out)


def usable(out, need=()):
    """True only when `out` is a json document that actually parses and carries what we read from it.

    SIZE IS NOT VALIDITY. Both reuse paths accepted any non-empty file, so a truncated or half-written
    json -- a file containing the single character "{" passes `getsize() > 0` -- was reused forever as
    a cached result, and a scoring failure became a silent exclusion from the evaluated population.
    A file that does not parse is deleted so the next call rescores rather than inheriting it.
    """
    try:
        with open(out) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        try:
            os.replace(out, f"{out}.invalid")      # kept for inspection, out of the reuse path
        except OSError:
            pass
        return False
    if not isinstance(d, dict):
        return False
    return all(k in d for k in need)


def run(model, ref, out, extra=None, chain_map=None, timeout=TIMEOUT):
    """invoke(), but skip if `out` already holds a result, and retry once on failure.

    `extra` appends further OST flags -- some callers want --lddt rather than DockQ, and dropping it
    would silently change what they measure. --rigid-scores is stripped, because it aborts on
    interfaces with fewer than three superposable points and takes the whole json down with it.

    THE RETRY: OST's automatic chain mapper crashes on some multi-chain natives (8ts0). Falling back
    to an explicit identity map rescues those, and is a no-op when the auto mapper would have worked.
    """
    if os.path.exists(out) and usable(out):
        return True
    if invoke(model, ref, out, extra=extra, chain_map=chain_map, timeout=timeout):
        return True
    if chain_map:                                  # caller already chose a map; nothing to fall back to
        return False
    ids = _identity_chain_map(ref)
    return bool(ids) and invoke(model, ref, out, extra=extra, chain_map=ids, timeout=timeout)


def _identity_chain_map(ref):
    """['A:A','B:B',...] from the reference's chain names, for the -c retry."""
    try:
        import gemmi
        st = gemmi.read_structure(str(ref))
        st.setup_entities()
        return [f"{ch.name}:{ch.name}" for ch in st[0]]
    except Exception:
        return []


def _chain_index(name):
    """Model chain name -> asym index, for indexing chain_pair_iptm. 'A'->0, 'AA'->26."""
    v = 0
    for c in name.split("-")[0]:
        v = v * 26 + (ord(c.upper()) - 64)
    return v - 1


def parse(out, conf_json=None):
    """Read an OST json into {"<refC1>,<refC2>": {dockq, model_chains, pair_iptm?}}.

    Returns {} rather than raising if the file is missing or truncated -- a crashed OST run leaves
    exactly that, and the caller needs to tell "no result" from "result of zero".
    """
    try:
        d = json.load(open(out))
    except Exception:
        return {}
    dq = d.get("dockq") or []
    res = {}
    for k, t in enumerate(d.get("dockq_interfaces") or []):
        ref_pair = f"{t[0]},{t[1]}"
        res[ref_pair] = {"dockq": dq[k] if k < len(dq) else None,
                         "model_chains": (t[2], t[3])}
    if conf_json:
        try:
            M = json.load(open(conf_json)).get("chain_pair_iptm")
            for v in res.values():
                i, j = (_chain_index(c) for c in v["model_chains"])
                if M and 0 <= i < len(M) and 0 <= j < len(M[i]):
                    v["pair_iptm"] = M[i][j]
        except Exception:
            pass
    return res


def interface(res, c1, c2):
    """in: a parse()/score() result and two REFERENCE chain names.  out: that interface's entry, or None

    Looked up in either order. FoldBench references can hold the complex twice, the second copy's chains
    named "<X>-2" (8fxv: A, B, A-2, B-2), and OST lays the model over whichever copy fits best -- so the
    interface A-B can come back as "A-2,B-2". That is the same interface, and it is accepted as such
    when both chains are from the same copy; a pair across copies ("A,B-2") is a different interface
    and is not. Raises if two copies both match, which OST's one-to-one mapping should never produce.
    References without copies (pxm2025) only ever take the first branch.
    """
    v = res.get(f"{c1},{c2}") or res.get(f"{c2},{c1}")
    if v:
        return v

    def base(name):
        head, _, tail = name.rpartition("-")
        return (head, tail) if head and tail.isdigit() else (name, "")

    hits = []
    for key, v in res.items():
        (b1, n1), (b2, n2) = (base(c) for c in key.split(","))
        if n1 == n2 and n1 and {b1, b2} == {c1, c2}:
            hits.append(v)
    if len(hits) > 1:
        raise ValueError(f"interface {c1},{c2} matches {len(hits)} reference copies")
    return hits[0] if hits else None


def score(model, ref, out, conf_json=None, extra=None, timeout=TIMEOUT):
    """run + parse. The single entry point callers should use."""
    if not run(model, ref, out, extra=extra, timeout=timeout):
        return {}
    return parse(out, conf_json=conf_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", default=None, help="summary_confidence json, to attach pair ipTM")
    A = ap.parse_args()
    r = score(A.model, A.ref, A.out, conf_json=A.conf)
    if not r:
        raise SystemExit(f"no result: OST produced nothing for {A.model}")
    for pair, v in sorted(r.items()):
        ip = f"  ipTM {v['pair_iptm']:.4f}" if v.get("pair_iptm") is not None else ""
        dq = "None" if v["dockq"] is None else f"{v['dockq']:.4f}"
        print(f"{pair}\tDockQ {dq}\tmodel {'/'.join(v['model_chains'])}{ip}")


if __name__ == "__main__":
    main()
