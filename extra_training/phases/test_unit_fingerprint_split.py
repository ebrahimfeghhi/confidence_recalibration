#!/usr/bin/env python
"""Gate: splitting unit_fingerprint into a CONTENT half and a code/loss/schema suffix changed NO
string anyone compares.

WHY THIS IS THE GATE THAT MATTERS. The fingerprint is the provenance guard. A label stores a
per-pose PERMUTATION of its OWN trunk's poses, so pairing a label with a different root's
same-named trunk trains pose i against pose j's label -- reproduced in review round 8 with a
fold_regen label and a fold trunk. The guard works by comparing the label's stored fingerprint
string against one recomputed now, so the ONLY acceptable refactor of that string is one that
leaves it byte-identical. This asserts that directly, against a literal of the pre-split format,
rather than trusting that the field order was preserved.

It also asserts the point of the split: the content half must be INDEPENDENT of code_fp and
loss_fp (so a code edit cannot invalidate it) and DEPENDENT on every file it covers (so a rebuild
does).

CPU ONLY, on temporary files it makes itself. Nothing here reads the real cache.

    /home/ebrahim/Protenix/.venv/bin/python -I phases/test_unit_fingerprint_split.py
"""
import gzip
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_j_build_cache as BC                                        # noqa: E402


def _fixture(d, pdb="1abc", trunk_bytes=b"trunk-v1"):
    os.makedirs(os.path.join(d, "cif"), exist_ok=True)
    os.makedirs(os.path.join(d, "bio"), exist_ok=True)
    open(os.path.join(d, "cif", f"{pdb}.cif"), "w").write("data_1abc\n")
    with gzip.open(os.path.join(d, "bio", f"{pdb}.pkl.gz"), "wb") as fh:
        fh.write(b"bioassembly")
    idx = os.path.join(d, "indices.csv")
    open(idx, "w").write("pdb_id,cluster_id\n1abc,7\n")
    trunk = os.path.join(d, "trunk.npz")
    open(trunk, "wb").write(trunk_bytes)
    a = types.SimpleNamespace(mmcif_dir=os.path.join(d, "cif"),
                              bio_dir=os.path.join(d, "bio"), indices=idx)
    return a, pdb, trunk


def _pre_split(a, pdb, trunk_p, code_fp, loss_fp):
    """unit_fingerprint EXACTLY as it read before the split, copied in rather than imported."""
    return "|".join([
        f"schema={BC.SCHEMA}",
        f"code={code_fp}",
        f"loss={loss_fp}",
        f"cif={BC.sha(os.path.join(a.mmcif_dir, f'{pdb}.cif'))}",
        f"bio={BC.sha(os.path.join(a.bio_dir, f'{pdb}.pkl.gz'))}",
        f"idx={BC._indices_by_pdb(a.indices).get(pdb, 'absent')}",
        f"trunk={BC.sha(trunk_p)}",
    ])


def main():
    with tempfile.TemporaryDirectory() as d:
        a, pdb, trunk = _fixture(d)
        code_fp, loss_fp = "c0ffee00c0ffee00", "1oss1oss1oss1oss"

        want = _pre_split(a, pdb, trunk, code_fp, loss_fp)
        got = BC.unit_fingerprint(a, pdb, trunk, code_fp, loss_fp)
        assert got == want, f"format changed:\n  pre-split {want}\n  now       {got}"
        assert len(got.split("|")) == 7, got
        assert [p.split("=", 1)[0] for p in got.split("|")] == \
            ["schema", "code", "loss", "cif", "bio", "idx", "trunk"], got
        print(f"OK composed string byte-identical to the pre-split format\n   {got}")

        # THE MEMO MUST RECOMPOSE THE SAME STRING.
        content = BC.unit_content_fingerprint(a, pdb, trunk)
        assert BC.unit_fingerprint(a, pdb, trunk, code_fp, loss_fp, content=content) == want
        assert content == "|".join(want.split("|")[3:]), (content, want)
        print(f"OK content= recomposes identically\n   {content}")

        # AND MUST BE INDEPENDENT OF code/loss -- that is the entire saving.
        assert BC.unit_content_fingerprint(a, pdb, trunk) == content
        other = BC.unit_fingerprint(a, pdb, trunk, "deadbeefdeadbeef", "f00df00df00df00d",
                                    content=content)
        assert other != want and other.split("|")[3:] == want.split("|")[3:], other
        print("OK the content half does not move when code_fp/loss_fp do")

        # AND MUST MOVE WHEN ANY COVERED FILE DOES.
        moved = []
        open(os.path.join(d, "trunk.npz"), "wb").write(b"trunk-v2-different-length")
        if BC.unit_content_fingerprint(a, pdb, trunk) != content:
            moved.append("trunk")
        open(os.path.join(d, "cif", f"{pdb}.cif"), "w").write("data_1abc\n# edited\n")
        if BC.unit_content_fingerprint(a, pdb, trunk) != content:
            moved.append("cif")
        with gzip.open(os.path.join(d, "bio", f"{pdb}.pkl.gz"), "wb") as fh:
            fh.write(b"bioassembly-edited")
        if BC.unit_content_fingerprint(a, pdb, trunk) != content:
            moved.append("bio")
        assert moved == ["trunk", "cif", "bio"], moved
        print("OK the content half moves when the trunk, the cif or the bioassembly does")

        # THE INDICES ROWS TOO, which are cached by path -- a NEW path, so the cache cannot mask it.
        a2, pdb2, trunk2 = _fixture(os.path.join(d, "second"))
        base2 = BC.unit_content_fingerprint(a2, pdb2, trunk2)
        open(a2.indices, "w").write("pdb_id,cluster_id\n1abc,9\n")
        BC._indices_by_pdb.cache_clear()
        assert BC.unit_content_fingerprint(a2, pdb2, trunk2) != base2
        print("OK the content half moves when the structure's indices rows do")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
