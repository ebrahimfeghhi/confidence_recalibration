#!/usr/bin/env python
"""--crop-tokens: the guards, the mask bookkeeping, and the atom-mask derivation.

WHAT THIS CAN AND CANNOT CHECK. The loader and _crop_masks are closures inside phase_k_train's
main function, so they cannot be imported; the real slicing is only observable from a run. This
covers the parts that do not need a card:

  1  THE DERIVATION. keep_atom induced from keep_tok through atom_to_token_idx, including the
     out-of-range entries that must not index the mask.
  2  THE __meta TRAP. build_union_crop_masks writes <pdb>, <pdb>__n AND __meta. A loader that
     filters only __n counts __meta as a structure -- which is exactly why phase_k_eval reports
     one more structure than it has. The trainer's loader must filter both.
  3  THE GUARDS, pinned in source: mutual exclusion with --crop-margin, a missing file, the
     --minimal-items requirement, a per-structure missing mask being FATAL, the token-count
     mismatch check, and the bookkeeping gates that would otherwise silently skip the crop csv.

Run:  python -I run_logging/test_crop_tokens.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "phases", "phase_k_train.py")
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


# ---- 1. the atom-mask derivation, as the trainer does it --------------------------------
def keep_atom_from(keep_tok, a2t, n_tok):
    inr = (a2t >= 0) & (a2t < n_tok)
    ka = np.zeros(a2t.shape[0], dtype=bool)
    ka[inr] = keep_tok[a2t[inr]]
    return ka


N_TOK = 6
kt = np.array([True, False, True, True, False, False])
# -1 is the padding the caches use for atoms with no token; 99 is out of range.
a2t = np.array([0, 0, 1, 2, 3, 3, -1, 99, 5])
ka = keep_atom_from(kt, a2t, N_TOK)
check("kept tokens' atoms are kept", list(ka[[0, 1, 3, 4, 5]]) == [True, True, True, True, True],
      f"{list(ka.astype(int))}")
check("dropped tokens' atoms are dropped", not ka[2] and not ka[8])
check("out-of-range atom indices are dropped, not used to index",
      not ka[6] and not ka[7], "-1 and 99 both excluded")
check("whole tokens only: every atom of a kept token comes along",
      bool(ka[0] and ka[1]), "token 0 has two atoms")

# ---- 2. the __meta trap ----------------------------------------------------------------
files = ["7abc", "7abc__n", "8xyz", "8xyz__n", "__meta"]
naive = [k for k in files if not k.endswith("__n")]
fixed = [k for k in files if not (k.endswith("__n") or k == "__meta")]
check("a filter on __n alone miscounts __meta as a structure", naive == ["7abc", "8xyz", "__meta"],
      f"{naive}")
check("filtering both gives the real structure count", fixed == ["7abc", "8xyz"], f"{fixed}")

src = open(TRAIN).read()
check("the trainer's loader filters __n AND __meta",
      'if not (k.endswith("__n") or k == "__meta")' in src)

# ---- 3. the guards, pinned ------------------------------------------------------------
for name, frag in [
    ("mutually exclusive with --crop-margin",
     "if a.crop_tokens and a.crop_margin > 0:"),
    ("a missing mask file is refused at startup",
     "if a.crop_tokens and not os.path.exists(a.crop_tokens):"),
    ("--minimal-items is required",
     "if a.crop_tokens and not a.minimal_items:"),
    ("a per-structure missing mask is FATAL",
     "--crop-tokens has no mask for"),
    ("a token-count mismatch against the cached trunk is refused",
     "was built against a different cache"),
    ("the crop csv is written under --crop-tokens too",
     "if a.crop_margin > 0 or a.crop_tokens:"),
    ("--force-gt absence is reported, not assumed",
     "NO --force-gt, so interface coverage is NOT guaranteed"),
]:
    check(name, frag in src)

check("BOTH crop call sites dispatch on the flag",
      src.count("_crop_masks(pdb, zc) if a.crop_tokens") == 1
      and src.count("_crop_masks(pdb, _zc) if a.crop_tokens") == 1,
      "prefetch worker and inline path")
check("the three bookkeeping gates all include crop_tokens",
      src.count("if a.crop_margin > 0 or a.crop_tokens:") == 3,
      f"{src.count('if a.crop_margin > 0 or a.crop_tokens:')} gate(s)")
check("it is a guard field on both sides",
      src.count('"crop_tokens": _CROP_TOK_DIGEST,') == 2,
      f"{src.count(chr(34) + 'crop_tokens' + chr(34) + ': _CROP_TOK_DIGEST,')} site(s)")
# THE GUARD MUST COMPARE CONTENTS, NOT THE FILENAME. It recorded basename() first, so rebuilding
# the mask with a different margin or forced-pair rule under the same name resumed as "unchanged"
# while the entire training input distribution had moved. Found in review.
check("the guard does NOT compare the filename",
      'os.path.basename(a.crop_tokens)' not in src)
check("the digest is over the per-structure MASKS and the provenance blob",
      "_ctd.update(np.packbits(np.asarray(_z0[_k0]).astype(bool)).tobytes())" in src
      and '_ctd.update(np.asarray(_z0["__meta"]).tobytes())' in src)
check("the digest is computed at STARTUP, before the guard dict is built",
      src.index("_CROP_TOK_DIGEST = _ctd.hexdigest()")
      < src.index('"crop_tokens": _CROP_TOK_DIGEST,'))
check("the banner reports the digest so a run says which mask it used",
      'content digest {_CROP_TOK_DIGEST}' in src)
# THE BUG THIS ALMOST SHIPPED: the guard lines were rewritten to _CROP_TOK_DIGEST while the patch
# that DEFINES it had failed its own assertion, so the trainer referenced an undefined name and
# would have NameError'd at startup. Assert the definition exists and precedes both uses.
check("_CROP_TOK_DIGEST is actually DEFINED, not just referenced",
      '_CROP_TOK_DIGEST = ""' in src and "_CROP_TOK_DIGEST = _ctd.hexdigest()" in src)
check("pre-flag bundles get a migration default",
      '"crop_tokens": ""' in src)

# ---- 4. the builder's side -------------------------------------------------------------
bsrc = open(os.path.join(ROOT, "run_logging", "build_union_crop_masks.py")).read()
check("--force-gt exists and requires --roles",
     '"--force-gt"' in bsrc and "--force-gt needs --roles" in bsrc)
check("the forced set is the BARE contact tokens (no margin on it)",
      "def gt_contact_tokens" in bsrc and "No margin: this set exists only" in bsrc)
check("the builder records how many tokens forcing added",
      "n_forced" in bsrc and "nocontact,\n                                 n_forced]" in bsrc)
check("coverage after forcing is asserted, not assumed",
      "arithmetically impossible" in bsrc)
# FINDING 2: the forced set must cover the pair the TRAINER supervises, which it resolves per
# pose through OST, not the static pool mapping -- they disagree on 52 of 368 structures.
check("--ost-pairs exists and needs --force-gt and --fold",
      '"--ost-pairs"' in bsrc and "only affects the forced ground-truth set" in bsrc
      and "needs --fold to find each pose" in bsrc)
check("the forced set unions BOTH mappings, not just the static one",
      "_pairs_use |= _op" in bsrc)
# THE FALLBACK IS GONE, not merely reported. "Named but accepted" still shipped a mask whose
# guarantee did not hold for the named structures, which is what the review refused.
check("there is NO static-pair fallback left to accept",
      "ost_unresolved" not in bsrc and "no fallback any more" in bsrc)
check("the builder reports how many structures the static mapping would have got wrong",
      "does NOT name" in bsrc)
# EXACT OST COVERAGE, raised in review of e0ee07b5: 49 of 50 poses resolved is NOT covered,
# because the unexamined pose can reference a pair absent from the forced set.
check("ANY unresolved pose is fatal, not tolerated",
      "could not resolve every expected pose" in bsrc
      and "_fails or _nok != _want_poses" in bsrc)
check("each failure carries its (seed, sample, why)",
      'failures.append((int(sd), int(k), why' in bsrc
      and 'f"seed_dir: {type(_e).__name__}"' in bsrc)
check("a seed whose unit dir fails counts ALL its poses as failures, not zero",
      "for k in range(nsample)]" in bsrc)
check("--ost-pairs requires a real seed range, so it cannot examine zero poses",
      "needs --seed-base and --budget" in bsrc)
check("the mask metadata records force_gt and ost_pairs",
      "force_gt=bool(a.force_gt), ost_pairs=bool(a.ost_pairs)," in bsrc)

# THE LAUNCH GATE. A content digest proves the mask did not change; it cannot prove the mask was
# right to begin with, so the launcher reads the mask's own metadata.
lsrc = open(os.path.join(ROOT, "shell_scripts", "launch_unioncrop_dropout_l2.sh")).read()
check("the launcher points at the OST mask, not the static-only one",
      "union_crop_m5_gt_ost.npz" in lsrc and "MASK=training/full_train/union_crop_m5_gt.npz"
      not in lsrc)
check("the launcher verifies the mask's provenance, not just its existence",
      "check_crop_mask.py" in lsrc and "--require-force-gt" in lsrc
      and "--require-ost-pairs" in lsrc)
check("the launcher demands the expected structure and pose counts",
      "--structures 368" in lsrc and "--poses 50" in lsrc)
check("the documented build command includes --ost-pairs and --fold",
      "--force-gt --ost-pairs" in lsrc and "--fold training/fold" in lsrc)
check("--lean-readouts is actually in the launcher's argv",
      "--pae-only --lean-readouts" in lsrc)

csrc = open(os.path.join(ROOT, "run_logging", "check_crop_mask.py")).read()
check("the checker refuses a mask with no __meta at all",
      "has no __meta" in csrc)
check("the checker treats a missing force_gt/ost_pairs flag as a refusal",
      'not meta.get("force_gt")' in csrc and 'not meta.get("ost_pairs")' in csrc)
# THE POSE COUNT MUST BE THE EXAMINED ONE. __n[0] is the count that CONTRIBUTED a contact-defined
# crop and dips to 149 on valfull's one dissociated decoy; __n[3] is the count examined and is
# 150 everywhere. Checking __n[0] forced the launcher to waive the constraint on valfull, which
# was a weaker guarantee than it appeared. Raised in review.
check("the checker validates EXAMINED poses (__n[3]), not contributing (__n[0])",
      "int(v[3]) != a.poses" in csrc and "int(v[0]) != a.poses" not in csrc)
check("a dissociated decoy is reported, not refused -- unless it is most of them",
      "0.05 * _tot" in csrc and "too many to be dissociated decoys" in csrc)
check("the launcher now REQUIRES 53 structures and 150 poses on valfull",
      "--structures 53 --poses 150" in lsrc)

print()
if FAIL:
    print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("  all checks passed")
