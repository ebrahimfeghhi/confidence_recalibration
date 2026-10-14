# The external Protenix modifications

`protenix_local.patch` is `git diff HEAD` taken in the external Protenix checkout that produced
every pose and trunk in this repository. It is checked in because our own commit sha does not
identify it: the trunk, the diffusion module and the confidence head all live there, and two caches
built under the same `git_sha` can otherwise come from different inference regimes.

| field | value at capture |
|---|---|
| Protenix commit | `bbc26ef` |
| tree dirty | yes — this patch is the difference |
| patch size | 35,155 bytes |
| checkpoint | `~/checkpoint/protenix-v2.pt`, sha256 `8f931f9774a396b6…` |
| python / torch | 3.12.3 / 2.7.1+cu128 |

`runmeta.external_provenance()` records all of the above into every `run.json`, so a fold can be
traced to the exact code and weights that produced it. To reconstruct: check out `bbc26ef`, apply
this patch, and verify the checkpoint sha256.
