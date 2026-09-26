# Donor selection: crystal-free (A) vs crystal-defined interfaces (B)

Same pool (seeds 300000-300049), pose-to-pose DockQ, diversity filter, round-robin and cap; only the interface and chain definition differs. Donor DockQ is against the crystal at each crystal interface (evaluation). Produced by `analysis/crystal_frame_selection.py`.

## Donors per complex

| selection | complexes | donors | median | mean | 1-5 | 6-10 | 11-20 | 21-30 | 31-49 | 50 (cap) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old arm (crystal-mapped ipTM) | 123 | 3342 | 29 | 27.2 | 25 | 8 | 19 | 13 | 27 | 31 |
| A crystal-free | 123 | 3524 | 31 | 28.7 | 24 | 9 | 16 | 10 | 25 | 39 |
| B crystal-defined | 123 | 3221 | 24 | 26.2 | 26 | 9 | 19 | 14 | 25 | 30 |

## Donor quality per class (crystal interfaces)

### old arm (crystal-mapped ipTM)

| class | interfaces | any acceptable donor | median best donor DockQ | mean donor DockQ |
|:---|---:|---:|---:|---:|
| decoy | 89 | 68/89 | 0.351 | 0.062 |
| solved | 142 | 142/142 | 0.786 | 0.418 |
| nocov | 61 | 0/61 | 0.060 | 0.018 |

donor-interface pairs without a crystal score (mapping failed): 0

### A crystal-free

| class | interfaces | any acceptable donor | median best donor DockQ | mean donor DockQ |
|:---|---:|---:|---:|---:|
| decoy | 89 | 69/89 | 0.369 | 0.065 |
| solved | 142 | 142/142 | 0.786 | 0.417 |
| nocov | 61 | 0/61 | 0.057 | 0.019 |

donor-interface pairs without a crystal score (mapping failed): 0

### B crystal-defined

| class | interfaces | any acceptable donor | median best donor DockQ | mean donor DockQ |
|:---|---:|---:|---:|---:|
| decoy | 89 | 68/89 | 0.354 | 0.062 |
| solved | 142 | 142/142 | 0.768 | 0.403 |
| nocov | 61 | 0/61 | 0.058 | 0.019 |

donor-interface pairs without a crystal score (mapping failed): 0

## Overlap of A and B

- complexes with identical donor sets: 75 / 123
- donors in both: 2795 (A has 3524, B has 3221)
