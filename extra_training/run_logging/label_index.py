#!/usr/bin/env python
"""THE ONLY SANCTIONED WAY TO ADDRESS A ROW OF A LABEL NPZ.

ROW POSITION IS NOT POSE ID. A label npz holds n_pose rows and a `sample_index` array saying which
SAMPLE each row belongs to. Those are not the same number: measured 2026-11-19, 99.2% of both the
7,309 training labels and the 2,490 validation labels are permuted (only 30 of 4,000 training and
21 of 2,490 validation labels have identity row order). So row 2 is routinely sample 0.

WHAT IT ALREADY COST. run_logging/val_pae_split.py read the loss trajectory's `pose` column -- a
ROW index -- as a sample id and looked up DockQ with it, which mis-joined essentially every row and
silently scrambled the good/bad split of every held-out PAE number reported that day (shipped's
good-pose mean moved 3.0991 -> 3.1203 once fixed; the all-pose mean did not move at all, which is
exactly why the error was invisible). The same confusion had already produced two earlier bugs: the
rel target regressed pose 1 onto pose 4's true iTM, and --emit-iptm-rows wrote the pose index while
73.8% of its rows carried another sample's DockQ -- moving the aggregate by 0.0009, less than the
noise it was read against.

THAT IS THE TRAP THIS MODULE EXISTS TO CLOSE: a wrong row/sample join does not fail, it returns a
plausible number. Open-coding the lookup was the real defect -- it had been written out by hand in
four places -- so every reader now goes through here instead, and a new script that reaches for
`lab["sample_index"]` itself should be changed to call these.

WHEN LABELS ARE NEXT REBUILT, in a new versioned directory, write the rows IN POSE ORDER so row
and sample agree and the trap is gone at the source. That is not worth rebuilding 9,000 labels for
on its own; it is worth doing the next time they are rebuilt anyway.
"""
import numpy as np


def sample_index_array(lab):
    """The label's row -> sample map, as a list of python ints. Raises if absent."""
    sv = lab.get("sample_index") if hasattr(lab, "get") else None
    if sv is None:
        try:
            sv = lab["sample_index"]
        except Exception as e:                                   # noqa: BLE001
            raise KeyError(f"label has no sample_index ({type(e).__name__}); its rows cannot be "
                           f"addressed by pose id") from e
    return [int(x) for x in np.asarray(sv).reshape(-1)]


def sample_of_row(lab, row):
    """Which SAMPLE the given row position holds. Use when reading a stored row index."""
    sl = sample_index_array(lab)
    if not 0 <= int(row) < len(sl):
        raise IndexError(f"row {row} is outside a {len(sl)}-row label")
    return sl[int(row)]


def row_of_sample(lab, sample):
    """Which ROW holds the given sample. Use when you have a pose/sample id and need its label.

    RAISES ON A DUPLICATE rather than returning the first match: two rows claiming one sample means
    the file is not what its name says, and picking either would be a guess.
    """
    sl = sample_index_array(lab)
    hits = [i for i, s in enumerate(sl) if s == int(sample)]
    if not hits:
        raise KeyError(f"sample {sample} is not in this label (samples {sl})")
    if len(hits) > 1:
        raise ValueError(f"sample {sample} appears in rows {hits} of one label")
    return hits[0]


def pae_target_row(lab):
    """The row to use for a PAE TARGET -- any row, so 0.

    WHY THIS IS SAFE AND WHY IT IS STILL A NAMED FUNCTION. There is one ground-truth structure per
    seed, and the per-pose rows differ only by symmetric SIDE-CHAIN atom swaps; PAE is built from
    the per-token REPRESENTATIVE atoms, which that permutation never touches. Verified on the PAE
    representative atoms over all 8,981 training labels by check_pae_label_shared.py (0 moved) and
    re-verified 2026-11-19 over 3,000 training and all 2,490 validation labels: identical, worst
    displacement 0.000000 A, coordinate_mask identical too.

    So row 0 is correct for every pose, which is what lets a PAE loss be computed before the
    pose-to-sample join is known. It is a function rather than a bare 0 so the justification has
    one home and a future change to the cache format has one place to break.
    """
    return 0
