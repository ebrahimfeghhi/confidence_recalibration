# Self-templating vs free + compute-matched

dataset `pxm2025`, experiment `self_template_fixed_50_seed` -- 292 interfaces counted; 0 excluded as unfinished; 0 complexes not yet compute-matched. Acceptable = DockQ >= 0.23 for the selected (top-ipTM) pose.

## Per interface

| class | interfaces | free + matched acceptable | self-template acceptable | free + matched mean DockQ | self-template mean DockQ | self-template only | free + matched only | McNemar p |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| decoy | 89 | 6/89 | 20/89 | 0.085 | 0.167 | 17 | 3 | 0.00258 |
| solved | 142 | 141/142 | 137/142 | 0.688 | 0.687 | 1 | 5 | 0.219 |
| nocov | 61 | 0/61 | 1/61 | 0.026 | 0.036 | 1 | 0 | 1 |

## Per complex (each complex weighted equally)

| class | complexes | free + matched acceptable rate | self-template acceptable rate | difference, 95% CI | free + matched mean DockQ | self-template mean DockQ | difference, 95% CI |
|:---|---:|---:|---:|---:|---:|---:|---:|
| decoy | 45 | 0.070 | 0.219 | [+0.015, +0.281] | 0.112 | 0.180 | [-0.022, +0.158] |
| solved | 73 | 0.986 | 0.945 | [-0.110, +0.014] | 0.648 | 0.648 | [-0.034, +0.038] |
| nocov | 22 | 0.000 | 0.045 | [+0.000, +0.136] | 0.025 | 0.049 | [+0.003, +0.057] |

95% intervals: cluster bootstrap over complexes, 10000 resamples, seed 0. McNemar: exact, two-sided, on discordant interfaces (treats interfaces as independent).
