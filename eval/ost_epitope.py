#!/usr/bin/env python
"""How much of a reference interface's EPITOPE a pose recovers. Runs under the OST binary.

SUMMARY
    Epitope recall for one antibody-antigen chain pair: of the antigen residues the antibody touches
    in the crystal, the share the pose also touches. Antigen side only -- the paratope is excluded on
    purpose, because the antibody docks through its CDRs whether or not the pose is right, so counting
    both sides reports a correct paratope as partial credit for a wrong epitope.

    EVERYTHING STRUCTURAL IS OST's. `Scorer` cleans the structures, computes the chain mapping, the
    5 A all-atom residue contacts and the target-to-model residue alignment. This file adds two things
    OST has no entry point for: it selects ONE chain pair (not the whole complex) and keeps ONE SIDE of
    it. OST's own IPS recall is the closest built-in and does neither -- it pools every interface and
    both of their sides.

    WHY THE ALIGNMENT IS NEEDED. `contact_scorer.cent1.contacts` is in crystal numbering and
    `cent2.contacts` in pose numbering; the crystal has unmodelled residues, so the two do not share a
    residue index. Both are mapped into alignment-column space before intersecting, exactly as OST's
    own `_InterfaceScores` does it (contact_score.py), so a recall here means the same thing as the
    ICS/IPS scores do.

    UNDEFINED, NOT ZERO. A pair whose chains never touch in the crystal has no epitope to recover, and
    a pair OST's chain mapping does not cover cannot be read at all. Both write an empty recall with a
    reason, never a 0.0 -- a zero would be indistinguishable from "docked somewhere else entirely".

    THE CHAIN MAPPING IS GIVEN, NOT SEARCHED. Each row carries the pose's own map from the project's
    chain_maps.csv, the one selection and DockQ already use, so the epitope is read at the same chain
    correspondence as the DockQ reported next to it. Letting OST search for its own would also expose
    an upstream bug: its naive QS-score mapper raises `IndexError: arrays used as indices` on some
    complexes (9ena), losing the interface entirely.

    MIN PEPTIDE LENGTH 4, not Scorer's default of 6, matching eval/ost_dockq.py. Antibody-antigen sets
    contain short peptide antigens, and at the default they are dropped before scoring -- the chain is
    simply absent from the mapping, so the interface reads as unmappable rather than as short.

INPUTS
    --work: tsv, no header -- key, model cif, reference cif, reference antibody chain, reference
            antigen chain, chain mapping as json ({reference chain: pose chain}, "" to let OST
            search). Rows sharing a (model, reference, mapping) reuse one Scorer, so the file is
            read sorted.

OUTPUTS
    --out:  tsv, no header -- key, recall, epitope residues in the crystal, of those recovered,
            epitope residues in the pose, note, chain mapping used ("given" the project's, or "ost"
            its own search), then the crystal and pose epitopes themselves as comma-separated
            ALIGNMENT COLUMN indices. recall and the counts are empty when the note is set.

    WHY COLUMN INDICES. Poses of one complex share a sequence, so each aligns to the fixed crystal the
    same way, and a column index therefore names the same residue in every pose of that interface.
    That makes pose epitopes comparable to EACH OTHER, which residue numbers would not be -- they are
    each structure's own. The crystal column set is written on every row as a check: it depends only
    on the target and the alignment, so if it ever differs between two poses of one interface the
    alignment was not stable and the pose sets must not be compared.

EXAMPLE
    ost_epitope.py --work work.tsv --out recall.tsv

FUNCTIONS
    pair_contacts():  a ContactEntity and an ordered chain pair -> its contacts, orientation fixed
    epitope_recall(): a Scorer and the reference chain pair -> (recall, n_crystal, n_both, n_pose)
    main():           walk the worklist, one Scorer per distinct (model, reference)
"""
import argparse
import csv
import json
import sys
import traceback

from ost import io
from ost.mol.alg.scoring import Scorer

# eval/ost_dockq.py's values, so an epitope and a DockQ of the same pose see the same chains
MIN_PEP_LENGTH = 4
MIN_NUC_LENGTH = 4


def usable_mapping(probe, mapping):
    """in: a Scorer (any, same structures), the project's {reference chain: pose chain}.
    out: that mapping restricted to the chains OST KEPT, or {} if either wanted chain is gone

    OST cleans both structures before mapping and drops what it will not score -- ligands, waters,
    chains under the length minimums. A custom mapping naming any dropped chain is refused outright
    ('Target chain "D" is not available'), which silently turned 38 scorable pxm2025 interfaces into
    errors when the full map was passed through. Filtering keeps the mapping injective because a
    subset of an injective map is injective.
    """
    trg = {ch.GetName() for ch in probe.chain_mapper.target.chains}
    mdl = {ch.GetName() for ch in probe.model.chains}
    return {k: v for k, v in mapping.items() if k in trg and v in mdl}


def pair_contacts(cent, pair):
    """in: a ContactEntity, an ordered (chain 1, chain 2).  out: its contact set, in THAT order

    ContactEntity stores each interface under one of its two orderings, so a caller that asks for the
    other one gets the contacts back with the tuples reversed. Returning them unreversed would silently
    swap which side is the antigen.
    """
    if pair in cent.contacts:
        return cent.contacts[pair]
    flipped = (pair[1], pair[0])
    if flipped in cent.contacts:
        return {(c[1], c[0]) for c in cent.contacts[flipped]}
    return set()


def epitope_recall(scorer, ref_ab, ref_ag):
    """in: a Scorer, the reference antibody and antigen chain names.
    out: (recall, crystal epitope size, recovered, pose epitope size, crystal columns, pose
         columns) or (None, note)

    The chain pair is ordered (antibody, antigen) in both structures, so index 1 of every mapped
    contact is the antigen side and no further bookkeeping decides which side is which.
    """
    flat = scorer.mapping.GetFlatMapping()
    if ref_ab not in flat or ref_ag not in flat:
        return None, "chain pair not in OST chain mapping"
    cs = scorer.contact_scorer
    trg_pair = (ref_ab, ref_ag)
    mdl_pair = (flat[ref_ab], flat[ref_ag])
    ref_contacts = pair_contacts(cs.cent1, trg_pair)
    if not ref_contacts:
        return None, "chains do not contact in the crystal"
    mdl_contacts = pair_contacts(cs.cent2, mdl_pair)
    ab_aln = cs.alns[(trg_pair[0], mdl_pair[0])]
    ag_aln = cs.alns[(trg_pair[1], mdl_pair[1])]
    # column space: GetPos(0, i) is the crystal's residue i, GetPos(1, i) the pose's
    crystal = {ag_aln.GetPos(0, c[1]) for c in ref_contacts}
    pose = {ag_aln.GetPos(1, c[1]) for c in mdl_contacts}
    _ = ab_aln  # the antibody alignment is unused: only the antigen side is counted
    both = crystal & pose
    return (len(both) / len(crystal), len(crystal), len(both), len(pose),
            sorted(crystal), sorted(pose)), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()

    rows = list(csv.reader(open(A.work), delimiter="\t"))
    loaded, scorer, used = None, None, ""
    with open(A.out, "w") as fh:
        w = csv.writer(fh, delimiter="\t")
        for key, model, ref, ref_ab, ref_ag, mapping in rows:
            try:
                if loaded != (model, ref, mapping):
                    m = io.LoadMMCIF(model, fault_tolerant=True)
                    t = io.LoadMMCIF(ref, fault_tolerant=True)
                    kw = dict(min_pep_length=MIN_PEP_LENGTH, min_nuc_length=MIN_NUC_LENGTH)
                    scorer, used = Scorer(m, t, **kw), "ost"
                    if mapping:
                        keep = usable_mapping(scorer, json.loads(mapping))
                        # a given mapping is preferred, but OST can still refuse it on chem-group
                        # grounds; its own search then stands in rather than losing the interface
                        if ref_ab in keep and ref_ag in keep:
                            try:
                                scorer, used = Scorer(m, t, custom_mapping=keep, **kw), "given"
                            except Exception:
                                traceback.print_exc()
                    loaded = (model, ref, mapping)
                got, note = epitope_recall(scorer, ref_ab, ref_ag)
            except Exception:
                traceback.print_exc()
                got, note = None, "OST raised"
                loaded = None
            if got is None:
                w.writerow([key, "", "", "", "", note, used, "", ""])
            else:
                recall, n_crystal, n_both, n_pose, cry_cols, pose_cols = got
                w.writerow([key, f"{recall:.4f}", n_crystal, n_both, n_pose, "", used,
                            ",".join(map(str, cry_cols)), ",".join(map(str, pose_cols))])
            fh.flush()
            print(f"{key}\t{'' if got is None else round(got[0], 3)}\t{used}\t{note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
