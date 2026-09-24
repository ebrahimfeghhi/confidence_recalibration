# Self-templating without the crystal

Protenix gives many poses for one antibody–antigen pair. ipTM confidently picks a wrong one while good
poses sit unselected in the same pool. This picks ~50 structurally different poses, feeds each back in as
a template, and refolds. **Nothing in the selection path reads the native structure.**

Result on the reference-free run: **21 of 91 confident-decoy interfaces rescued, 0 broken.**

## The pipeline

Run in this order. Steps 1–6 never touch ground truth; step 8 is the only place the crystal appears.

| | file | what it does |
|---:|:---|:---|
| 0 | `build_free_queue.py` | write a queue of untemplated folds |
| 0 | `call_folder.py` | launch workers on any queue, one tmux session per GPU |
| 0 | `folder.py` | fold a queue with the checkpoint loaded once, score each fold |
| 0 | `scorer.py` | DockQ one fold's poses, print csv rows |
| 1 | `anchor_relabel.py` | name every pose's chains after one chosen pose, so "interface A–B" means the same thing in all of them |
| 2 | `pose_interfaces.py` | decide which chain pairs are interfaces, from contacts in the poses |
| 3 | `fetch_spurious_interface_iptm.py` | ipTM for the interfaces the predictions declare and the reference lacks |
| 4 | `pose_to_pose_dockq.py` | N×N symmetric pose-to-pose DockQ per interface |
| 5 | `structural_diversity_filter.py` | sort by ipTM, keep a pose only if unlike everything kept, round-robin, cap 50 |
| 6 | `build_templates.py` | turn each donor pose into a template Protenix can load |
| 7 | `preflight.py` | check ids, chains and a3m hits before spending GPU time |
| 8 | evaluation | pooled top-ipTM selection against the free-pool baseline, per class |

Shared: `pool_index.py` reads the index csvs, `contacts.py` decides whether two chains touch,
`watch.sh` babysits a running sweep, `hold_gpus.py` occupies cards while one is paused.

## Where the data lives

`experimental_data/` here is an **empty skeleton** showing the layout. Real data is not in the repo —
it is gigabytes — and lives at the same paths under the repo root, all of them defined in `paths.py`
and overridable by an environment variable of the same name:

    ANCHOR_DIR=/data/anchor python pose_interfaces.py --out /tmp/x

```
experimental_data/
  free/                             the untemplated pose pool: step 0's output, what everything selects from
    8ror/300000-a1b2c3d4e5/         one directory per fold, <seed>-<run_id>
      predictions/                  8ror_sample_0.cif ... and the confidence jsons
  anchor_frame/                     steps 1-3
    chain_maps_anchor.csv           per pose: {anchor_chain: pose_chain}
    chain_roles_anchor.csv          per interface: which anchor chains are antibody and antigen
    chain_roles_all.csv             the above plus the predicted-only pairs
    pose_interfaces.csv             every contacting pair, with have_gt
    spurious_pool.csv               ipTM for the predicted-only pairs
    anchor_manifest.json            which complexes this covers, and the anchor pose chosen for each
  matrices/                         step 4, crystal frame
  matrices_anchor_all/              step 4, anchor frame
  matrices_spur/                    step 4, predicted-only interfaces
  reffree_50/                       one experiment
    queue.tsv                       step 5: one line per fold to run
    ns/mmcif/<tid>.cif              step 6: each donor pose as a template
    ns/jobs/<iface>_<tag>/          step 6: job.json, chosen.json, a3m hits
    claims/                         one empty directory per claimed unit; the locking mechanism
    out/8w5d/430000-.../            step 7: the folds
    results.csv                     scores, and the to-do list
```

## Two things worth knowing before changing anything

**The csv is the to-do list.** `folder.py` skips any `(interface, seed, arm)` already in `results.csv`
and never looks at the filesystem, so the two can drift. `folder.py --verify` cross-checks them.

**A fold is identified by (pose, seed, pinned chains), not by its tag.** The tag is a label. When two
interfaces of one complex both used tag `K00`, they silently swapped donors — which is why every row
records `donor_pose` and `tmpl_chains`.
