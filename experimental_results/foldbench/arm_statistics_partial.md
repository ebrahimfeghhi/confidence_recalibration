# Self-templating vs free + compute-matched

dataset `foldbench`, experiment `self_template_fixed_50_seed` -- 58 interfaces counted; 0 excluded as unfinished; 67 complexes not yet compute-matched. Acceptable = DockQ >= 0.23 for the selected (top-ipTM) pose.

## Per interface

| class | interfaces | free + matched acceptable | self-template acceptable | free + matched mean DockQ | self-template mean DockQ | self-template only | free + matched only | McNemar p |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| decoy | 28 | 0/28 | 5/28 | 0.068 | 0.146 | 5 | 0 | 0.0625 |
| solved | 17 | 16/17 | 12/17 | 0.568 | 0.482 | 0 | 4 | 0.125 |
| nocov | 13 | 1/13 | 0/13 | 0.035 | 0.017 | 0 | 1 | 1 |

## Per complex (each complex weighted equally)

| class | complexes | free + matched acceptable rate | self-template acceptable rate | difference, 95% CI | free + matched mean DockQ | self-template mean DockQ | difference, 95% CI |
|:---|---:|---:|---:|---:|---:|---:|---:|
| decoy | 22 | 0.000 | 0.182 | [+0.045, +0.364] | 0.070 | 0.150 | [+0.008, +0.165] |
| solved | 13 | 0.923 | 0.692 | [-0.462, +0.000] | 0.531 | 0.458 | [-0.168, +0.002] |
| nocov | 9 | 0.111 | 0.000 | [-0.333, +0.000] | 0.043 | 0.015 | [-0.094, +0.007] |

95% intervals: cluster bootstrap over complexes, 10000 resamples, seed 0. McNemar: exact, two-sided, on discordant interfaces (treats interfaces as independent).
