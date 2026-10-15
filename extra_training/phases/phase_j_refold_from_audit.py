"""The refold list, derived from the domain audit rather than from a composition guess.

THE RULE, and why it is this narrow.

A query is refolded only when the correction is unambiguous: the role sits on a chain carrying ONE
variable domain, and the query omits native residues strictly INSIDE the stretch it already covers.
Filling an internal hole in a contiguous span needs no boundary decision -- the residues are known,
their positions are known, and nothing about where a domain starts or ends is in question.

Everything else is skipped, not guessed, for reasons the previous attempt demonstrated:

  * A single-chain construct (scFv) carries VH and VL on ONE deposited chain joined by a GS tether,
    and the splice point was being inferred from G/S composition. That inference failed both ways.
    It classified 8dpl's run 115-136 as tether although it opens with TVSS, the real end of VH
    (hmmscan: H 1-118, E=1.2e-58). And it classified pure tether as missing domain sequence for
    8cz8, 9ejz, 8vy9, 8pww, 9tuy and 32nz -- KGGG, GGG, GGGG, SGGG, SG, TTPP -- all of which were
    queued for RESTORATION. Refolding those would have written a worse input than the one it
    replaced. Under this rule they are excluded automatically: none is a one-domain chain.

  * The domain edge is not crisp to the residue. 28 scFv roles differ from the hmmscan boundary by
    one or two residues (short 241-241, outside 158-158, short 248-248 and so on). Rebuilding a
    query to the profile boundary would rewrite -- and force a refold of -- 28 inputs that are fine,
    on the strength of HMMER's opinion about an edge. Runs of one or two residues at a domain edge
    are therefore not treated as defects.

  * Query length separates architecture. 8qp6 chain B holds VH-CH1 and VL-CL, so hmmscan sees two
    variable domains and the heavy query's 104 residues "outside" its V domain are simply CH1 --
    a normal Fab heavy chain, not contamination. A V-only role runs ~105-135 residues; a V+C role
    runs ~180-230. `VC_MIN` is the split.

What the audit finds genuinely wrong, and what is skipped rather than refolded, is written to the
skip file with its measured reason, so the exclusions are auditable instead of asserted.

Input is `phase_j_domain_audit.py --out`. Nothing here re-derives an alignment.
"""
import argparse, collections, csv, os

VC_MIN = 180          # at or above this a role carries its constant domain too (see docstring)
EDGE = 2              # runs this short at a domain edge are the profile's boundary, not a defect
# A QUERY MUST ACTUALLY COVER ITS SPAN TO HAVE "HOLES" IN IT. A short peptide epitope aligns to a
# long chain by chance: 7tr4's ag2 is a 10-residue peptide whose residues difflib scatters from
# native 1 to 115, reported as 105 "missing" residues. Nothing is missing -- the epitope is ten
# residues long. Three antigen roles produced 437 of the list's 1372 residues that way, more than
# every heavy chain combined. Below this coverage the alignment is not a contiguous stretch with
# holes in it and no residue may be restored from it.
MIN_COVER = 0.80


def parse_runs(s):
    out = []
    for part in (s or "").split(";"):
        if part:
            a, b = part.split("-")
            out.append((int(a), int(b)))
    return out


def parse_domains(s):
    out = []
    for d in (s or "").split(";"):
        if d:
            t, sp = d.split(":")
            a, b = sp.split("-")
            out.append((t, int(a), int(b)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out", required=True, help="the refold list")
    ap.add_argument("--skip", required=True, help="roles with a defect this will NOT guess at")
    ap.add_argument("--accepted", default="", help="scFv roles accepted as boundary noise")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.audit)))
    refold, skip, accept = [], [], []

    for r in rows:
        if r.get("verdict") in ("no_cif", "no_polymer") or not r.get("chain"):
            continue
        nd = int(r.get("n_domains") or 0)
        gaps = parse_runs(r.get("dels"))
        qlen = int(r["qlen"])

        # --- one variable domain on the chain: an internal hole is unambiguous -------------------
        if nd <= 1:
            span_len = int(r["span_hi"]) - int(r["span_lo"]) + 1
            cover = qlen / span_len if span_len else 0.0
            if gaps and cover < MIN_COVER:
                skip.append(dict(
                    pdb=r["pdb"], role=r["role"], chain=r["chain"], qlen=qlen,
                    dom=f"span {r['span_lo']}-{r['span_hi']}",
                    missing_domain=r["dels"], n_missing_domain=sum(b - x + 1 for x, b in gaps),
                    reason=f"alignment unreliable: query covers {cover:.0%} of its span "
                           f"({qlen} residues over {span_len}); too sparse to call a hole"))
                continue
            if gaps:
                refold.append(dict(
                    pdb=r["pdb"], role=r["role"], chain=r["chain"], qlen=qlen,
                    span=f"{r['span_lo']}-{r['span_hi']}",
                    n_missing=sum(b - x + 1 for x, b in gaps),
                    missing=r["dels"], old_q=r["q"], corrected_q=r.get("corrected", ""),
                    reason="internal_omission_single_domain"))
            continue

        # --- two or more variable domains: decide, never patch -----------------------------------
        if qlen >= VC_MIN:
            accept.append(dict(pdb=r["pdb"], role=r["role"], chain=r["chain"], qlen=qlen,
                               reason="V+C architecture; residues past the V domain are constant"))
            continue

        doms = parse_domains(r.get("domains"))
        dl, dh = int(r["dom_lo"]), int(r["dom_hi"])
        other = [(x, b) for _t, x, b in doms if not (x == dl and b == dh)]
        nxt = min([x for x, b in other if x > dh], default=None)
        prv = max([b for x, b in other if b < dl], default=None)

        teth = []
        for x, b in parse_runs(r.get("outside_runs")):
            if b - x + 1 <= EDGE:
                continue
            if (nxt is not None and x > dh and b < nxt) or (prv is not None and b < dl and x > prv):
                teth.append((x, b))
        shortd = [(x, b) for x, b in parse_runs(r.get("short_runs")) if b - x + 1 > EDGE]

        if teth or shortd:
            skip.append(dict(
                pdb=r["pdb"], role=r["role"], chain=r["chain"], qlen=qlen,
                domains=r["domains"], dom=f"{r['dom_type']}:{dl}-{dh}",
                tether_in_query=";".join(f"{x}-{b}" for x, b in teth),
                n_tether=sum(b - x + 1 for x, b in teth),
                missing_domain=";".join(f"{x}-{b}" for x, b in shortd),
                n_missing_domain=sum(b - x + 1 for x, b in shortd),
                reason=("tether residues inside the query" if teth else "")
                       + ("; " if teth and shortd else "")
                       + ("domain residues absent from the query" if shortd else "")))
        else:
            accept.append(dict(pdb=r["pdb"], role=r["role"], chain=r["chain"], qlen=qlen,
                               reason=f"scFv, differs from the profile edge by <={EDGE}"))

    def dump(path, recs, cols):
        if not path:
            return
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for x in sorted(recs, key=lambda z: (z["pdb"], z["role"])):
                w.writerow(x)

    # EVERY ROW IS CHECKED, NOT ASSERTED: the corrected query must be exactly the old one with the
    # reported holes filled, so its length must be the span length and it must contain the old query
    # as a subsequence. A row that fails either is a bug in the alignment, not a refold to queue.
    bad = []
    for x in refold:
        lo, hi = (int(v) for v in x["span"].split("-"))
        want = hi - lo + 1
        c, o = x["corrected_q"], x["old_q"]
        it = iter(c)
        if len(c) != want:
            bad.append((x["pdb"], x["role"], f"corrected is {len(c)}, span is {want}"))
        elif len(c) != len(o) + x["n_missing"]:
            bad.append((x["pdb"], x["role"],
                        f"corrected {len(c)} != old {len(o)} + missing {x['n_missing']}"))
        elif not all(ch in it for ch in o):
            bad.append((x["pdb"], x["role"], "old query is not a subsequence of the corrected one"))
    if bad:
        print("STOP: %d corrected query/queries do not reconstruct:" % len(bad))
        for b in bad[:20]:
            print("   %-5s %-3s %s" % b)
        raise SystemExit(1)
    print("all %d corrected queries reconstruct (length == span, old query a subsequence)"
          % len(refold))

    dump(a.out, refold, ["pdb", "role", "chain", "qlen", "span", "n_missing", "missing",
                         "reason", "old_q", "corrected_q"])
    dump(a.skip, skip, ["pdb", "role", "chain", "qlen", "domains", "dom", "tether_in_query",
                        "n_tether", "missing_domain", "n_missing_domain", "reason"])
    dump(a.accepted, accept, ["pdb", "role", "chain", "qlen", "reason"])

    print("REFOLD  %d role(s) over %d structure(s), %d residue(s) -> %s"
          % (len(refold), len({x['pdb'] for x in refold}),
             sum(x["n_missing"] for x in refold), a.out))
    print("SKIP    %d role(s) over %d structure(s) -> %s"
          % (len(skip), len({x['pdb'] for x in skip}), a.skip))
    for x in sorted(skip, key=lambda z: z["pdb"]):
        print("   %-5s %-2s %s" % (x["pdb"], x["role"], x["reason"]))
    print("ACCEPT  %d scFv/V+C role(s) over %d structure(s)"
          % (len(accept), len({x['pdb'] for x in accept})))


if __name__ == "__main__":
    main()
