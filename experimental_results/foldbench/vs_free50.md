# self_template_fixed_50_seed

| class | pooling | interfaces | mean before | mean after | acceptable before | acceptable after | rescued | broken |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| decoy | templated only | 28 | 0.064 | 0.146 | 0/28 | 5/28 | 5 | 0 |
| decoy | free + templated | 28 | 0.064 | 0.148 | 0/28 | 5/28 | 5 | 0 |
| solved | templated only | 119 | 0.715 | 0.690 | 119/119 | 109/119 | 0 | 10 |
| solved | free + templated | 119 | 0.715 | 0.695 | 119/119 | 111/119 | 0 | 8 |
| nocov | templated only | 13 | 0.017 | 0.017 | 0/13 | 0/13 | 0 | 0 |
| nocov | free + templated | 13 | 0.017 | 0.017 | 0/13 | 0/13 | 0 | 0 |

  experiment: experimental_data/protenix/foldbench/self_template_fixed_50_seed
  interfaces scored: 160   excluded as unfinished: 0
  queue length: 3209 folds
