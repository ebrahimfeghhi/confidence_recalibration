#!/usr/bin/env python
"""Selected DockQ when the selector CANNOT SEPARATE the top poses, resolved without a sort.

THE PROBLEM. `max(poses, key=score)` returns whichever tied pose comes first in the list, so the
answer depends on row order -- which here follows the order a glob returned three shard files.
stored_iptm is recorded at FOUR DECIMAL PLACES, so ties are routine rather than exotic: the
shipped baseline moved by 0.0001 between two epochs that scored the identical candidates with the
identical fixed head, which is impossible and was purely the tie-break changing.

THE RULE. If a selector gives k poses the same best score it has expressed no preference among
them, so the honest number is the EXPECTED one under a uniform choice: the MEAN DockQ over the
tied set, and the FRACTION of the tied set that is acceptable. With no tie (k=1) both reduce
exactly to the old behaviour, so nothing changes where nothing was ambiguous.

WHY NOT A DETERMINISTIC TIE-BREAK, e.g. lowest (seed, sample). It would also be reproducible, and
it would be WRONG in a specific direction: it credits the selector with a choice it did not make.
On a tie between a 0.9 and a 0.1 pose, picking by index reports whichever the file order favours;
the expectation reports 0.5, which is what that selector is actually worth.

IT DISAGREES WITH phase_k_eval BY DESIGN, and the difference is small and must be stated.
phase_k_eval._pick breaks ties with max(v, key=(score, -seed, -sample)) -- also deterministic,
also order-independent, and therefore NOT the bug this fixes. On erb7nd's valfull pool the two
rules give a shipped baseline of 0.5831 (evaluator) and 0.5834 (expectation) -- 0.0003 apart, from
11 tied interfaces of 92. The expectation is the unbiased one; the evaluator's rule credits the
shipped head with a choice it did not make. The evaluator is left alone for now because its
numbers are the ones paper_results quotes, so changing it mid-project would silently move
published figures; a figure that uses this helper should say which rule it used.

EXACT COMPARISON IS INTENDED. Scores are compared with ==, not a tolerance: these are the stored
4dp values, and two poses that round to the same printed number ARE indistinguishable to anything
reading that column.
"""


def expected_pick(poses, score_idx, dq_idx, acc=0.23):
    """(expected DockQ, expected acceptable) over the poses tied at the best score.

    `poses` is any sequence of tuples; `score_idx` selects the score, `dq_idx` the DockQ.
    """
    if not poses:
        raise ValueError("expected_pick on an empty candidate set")
    best = max(p[score_idx] for p in poses)
    tied = [p[dq_idx] for p in poses if p[score_idx] == best]
    n = len(tied)
    return sum(tied) / n, sum(1.0 for d in tied if d >= acc) / n


def tie_report(by_iface, score_idx, keep=None):
    """(n_interfaces_with_a_tie, largest_tied_set) -- so a caller can SAY how much this mattered."""
    n_tied, worst = 0, 1
    for i in (keep if keep is not None else by_iface):
        v = by_iface[i]
        best = max(p[score_idx] for p in v)
        k = sum(1 for p in v if p[score_idx] == best)
        if k > 1:
            n_tied += 1
            worst = max(worst, k)
    return n_tied, worst


if __name__ == "__main__":
    # the behaviour this exists to pin, checked without any data on disk
    rows = [(0.5000, 0.9), (0.5000, 0.1), (0.4999, 1.0)]
    dq, ac = expected_pick(rows, 0, 1)
    assert abs(dq - 0.5) < 1e-12, dq
    assert abs(ac - 0.5) < 1e-12, ac
    assert expected_pick(list(reversed(rows)), 0, 1) == (dq, ac), "order changed the answer"
    one = [(0.7, 0.8), (0.1, 0.2)]
    assert expected_pick(one, 0, 1) == (0.8, 1.0), "no-tie case must reduce to max()"
    print("  ok: ties average, order does not matter, the no-tie case is unchanged")
