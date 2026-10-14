#!/usr/bin/env python
"""THE PXMeter DockQ scorer. Every PXMeter score in this project goes through here.

Companion to ost_dockq.py, and deliberately the same shape: both return
    {"<refC1>,<refC2>": {"dockq":..., "model_chains":..., "pair_iptm":...}}
so a caller can swap tools without changing anything downstream.

WHY TWO TOOLS AT ALL. OST is the FoldBench methodology and is what the older benchmark was scored
with; PXMeter is what the PXM-2025 sweep uses. Keeping both is deliberate -- comparing our numbers to
published FoldBench results requires OST, and re-scoring the whole PXM sweep with OST would cost days.
They are not interchangeable in a mixed analysis: pick one per comparison and say which.

CHAIN LETTERS. PXMeter keys `interface` on REFERENCE chain names and provides
`ref_to_model_chain_mapping` separately. ipTM (`chain_pair_iptm`) is indexed by MODEL asym order, so
the mapping is required to line the two up. Substituting one for the other once put 45% of this
project's interface measurements on the wrong chains.

Usage as a library:
    from pxm_dockq import score
    r = score(model_cif, ref_cif, out_json, conf_json=cj)

Usage from the shell:
    pxm_dockq.py --model m.cif --ref r.cif --out o.json [--conf summary_confidence.json]
"""
import argparse
import json
import os
import subprocess

# the repo root, from this file, so renaming the checkout needs no edit here
_REPO = str(__import__("pathlib").Path(__file__).resolve().parent.parent)

PXM = os.environ.get(
    "PXM_BIN",
    f"{_REPO}/envs/pxmeter/bin/pxm")
TIMEOUT = int(os.environ.get("PXM_TIMEOUT", "300"))


def usable(out):
    """True only when `out` parses as a json object.

    SIZE IS NOT VALIDITY: a file holding the single character "{" satisfies `getsize() > 0` and was
    reused as a cached score forever, turning a scoring failure into a silent exclusion. An
    unparseable file is moved aside so the next call rescores instead of inheriting it.
    """
    try:
        with open(out) as fh:
            d = json.load(fh)
        # A VALID-BUT-EMPTY OBJECT IS NOT A RESULT. `{}` parses and was accepted, so a crashed pxm
        # run cached as "scored" and became a silent exclusion. parse() reads `interface`.
        return isinstance(d, dict) and "interface" in d
    except (OSError, ValueError):
        try:
            os.replace(out, f"{out}.invalid")
        except OSError:
            pass
        return False


def run(model, ref, out, timeout=TIMEOUT):
    """Invoke pxm unless `out` already holds a result that parses."""
    if os.path.exists(out) and usable(out):
        return True
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    try:
        subprocess.run([PXM, "-r", ref, "-m", model, "-o", out],
                       capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return os.path.exists(out) and usable(out)


def _chain_index(name):
    """Model chain name -> asym index, for indexing chain_pair_iptm. 'A'->0, 'AA'->26."""
    v = 0
    for c in name.split("-")[0]:
        v = v * 26 + (ord(c.upper()) - 64)
    return v - 1


def parse(out, conf_json=None):
    """Read a PXMeter json into the same shape ost_dockq.parse returns.

    Returns {} on a missing or truncated file so callers can tell "no result" from "zero".
    """
    try:
        J = json.load(open(out))
    except Exception:
        return {}
    r2m = J.get("ref_to_model_chain_mapping", {})
    conf = None
    if conf_json:
        try:
            conf = json.load(open(conf_json)).get("chain_pair_iptm")
        except Exception:
            conf = None
    res = {}
    for key, v in (J.get("interface") or {}).items():
        if v.get("dockq") is None:
            continue
        c1, c2 = key.split(",")
        entry = {"dockq": float(v["dockq"])}
        if c1 in r2m and c2 in r2m:
            entry["model_chains"] = (r2m[c1], r2m[c2])
            if conf:
                i, j = _chain_index(r2m[c1]), _chain_index(r2m[c2])
                if 0 <= i < len(conf) and 0 <= j < len(conf[i]):
                    entry["pair_iptm"] = conf[i][j]
        res[key] = entry
    return res


def score(model, ref, out, conf_json=None, timeout=TIMEOUT):
    """run + parse. The single entry point callers should use."""
    if not run(model, ref, out, timeout=timeout):
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
        raise SystemExit(f"no result: PXMeter produced nothing for {A.model}")
    for pair, v in sorted(r.items()):
        ip = f"  ipTM {v['pair_iptm']:.4f}" if v.get("pair_iptm") is not None else ""
        mc = "/".join(v["model_chains"]) if v.get("model_chains") else "?"
        print(f"{pair}\tDockQ {v['dockq']:.4f}\tmodel {mc}{ip}")


if __name__ == "__main__":
    main()
