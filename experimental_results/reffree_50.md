# reffree_50

| class | pooling | interfaces | mean before | mean after | acceptable before | acceptable after | rescued | broken |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| decoy | templated only | 91 | 0.055 | 0.170 | 0/91 | 21/91 | 21 | 0 |
| decoy | free + templated | 91 | 0.055 | 0.151 | 0/91 | 17/91 | 17 | 0 |
| solved | templated only | 137 | 0.690 | 0.688 | 137/137 | 132/137 | 0 | 5 |
| solved | free + templated | 137 | 0.690 | 0.689 | 137/137 | 132/137 | 0 | 5 |
| nocov | templated only | 42 | 0.018 | 0.024 | 0/42 | 1/42 | 1 | 0 |
| nocov | free + templated | 42 | 0.018 | 0.024 | 0/42 | 1/42 | 1 | 0 |

  experiment: /home/ebrahim/tts_abag/experimental_data/reffree_50
  interfaces scored: 270   excluded as unfinished: 5
  queue length: 972 folds
