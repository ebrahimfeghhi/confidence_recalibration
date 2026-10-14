#!/usr/bin/env python
"""Score one fold's poses against the reference and print CSV rows on stdout.

SUMMARY
    Run as a subprocess by folder.py, once per interface. Writes nothing itself -- the caller captures
    stdout and appends it to results.csv. A separate process because DockQ scoring needs a different
    venv than folding, and because a pose the scorer chokes on kills a subprocess rather than the
    worker holding the model.

INPUTS
    --dir:     the fold's output directory. Every *_sample_*.cif under it gets scored.
    --iface:   which interface to score, as <pdb>_<C1><C2>, in REFERENCE chain letters. Reference
               letters are not model letters -- the scorers translate internally, per pose.
    --seed:    the fold's seed, copied through to the csv.
    --arm:     the queue tag, e.g. "A07" = the 8th donor selected for this complex ("A" names the
               sweep, 07 the donor's index; earlier sweeps used C## and K##). A LABEL only: the driver
               skips a unit whose (iface, seed, arm) is already in the csv, so a tag must be unique per
               complex, but it says nothing about what was templated.
    --donor:   the pose that was templated, or "-1" for a free fold. This, not --arm, is the
               provenance: a fold is identified by (pose, seed, pinned chains).
    --chains:  which chains of that donor were pinned, joined with "+". The same pose pinned to a
               different antigen copy is a DIFFERENT template and a different fold.
    --ref:     the reference structure to score against.
    --scorer:  ost or pxm. Default pxm. Both return the same shape, so the csv is identical either way.

OUTPUTS
    stdout: one line per pose, appended to results.csv by the caller:

        iface,seed,arm,donor_pose,tmpl_chains,sample,dockq,iptm
        8w5d_AB,430000,A07,/abs/8w5d_sample_4.cif,A+B+D,0,0.0092,0.2885

    sample is the diffusion sample index within the fold; dockq is against the reference; iptm is the
    model's own confidence for that chain pair.

EXAMPLE
    scorer.py --dir out/8w5d/430000-a1b2c3/8w5d_DC_A07 --iface 8w5d_AB --seed 430000 --arm A07 \
              --donor /abs/8w5d_sample_4.cif --chains A+B+D --ref cif/8w5d.cif

FUNCTIONS
    none -- a short script, read top to bottom
"""
import argparse
import glob
import os
import sys

# the repo root is this file's parent, so nothing here carries an absolute path; paths.py owns
# every location from that point on
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

sys.path.insert(0, str(paths.EVAL_DIR))
import ost_dockq  # noqa: E402
import pxm_dockq  # noqa: E402

ap = argparse.ArgumentParser()
for a in ("--dir", "--iface", "--seed", "--arm", "--donor", "--ref"):
    ap.add_argument(a, required=True)
ap.add_argument("--scorer", choices=("ost", "pxm"), default="pxm")
# which chains of the donor were pinned; part of a fold's identity, see the docstring
ap.add_argument("--chains", default="")
A = ap.parse_args()

# in:  --iface "<pdb>_<C1><C2>", REFERENCE chain letters (not model letters)
# out: the two chain names to score
chains = A.iface.split("_")[1]
if len(chains) != 2:
    raise SystemExit(f"--iface {A.iface}: expected <pdb>_<C1><C2> with single-letter chains")
c1, c2 = chains[0], chains[1]

mod = ost_dockq if A.scorer == "ost" else pxm_dockq
jdir = os.path.join(A.dir, f"{A.scorer}_json")
os.makedirs(jdir, exist_ok=True)

# in:  every *_sample_*.cif under --dir, with its confidence json
# out: one csv line per pose
# both scorers key results by REFERENCE chain names and translate to model names internally
for cif in sorted(glob.glob(f"{A.dir}/**/*_sample_*.cif", recursive=True)):
    k = cif.split("sample_")[-1].split(".")[0]
    conf = glob.glob(f"{A.dir}/**/*summary_confidence_sample_{k}.json", recursive=True)
    res = mod.score(cif, A.ref, os.path.join(jdir, f"{k}.json"),
                    conf_json=conf[0] if conf else None)
    # either order, or the same interface in a second copy of the reference (ost_dockq.interface)
    v = ost_dockq.interface(res, c1, c2)
    if not v or v.get("dockq") is None:
        continue          # this pose has no such interface -- a real condition, not an error
    ip = v.get("pair_iptm")
    print(f"{A.iface},{A.seed},{A.arm},{A.donor},{A.chains},{k},{float(v['dockq']):.4f},"
          f"{'' if ip is None else f'{float(ip):.4f}'}")
