# Every run trained on overfit30 and scored on val10

Built by scanning each run's own `args.json` and its `metric_train_ep*.json` files, keeping runs
whose `--manifest-selection` is overfit30 and whose eval has `n_interfaces == 18`. Grad probes are
excluded; `latest` is a symlink to the newest run and is dropped as a duplicate.

SHIPPED on val10 is DockQ 0.2909, frac 0.5556 (10 of 18). One interface is 0.0556 of frac, so a
single pose flipping moves that column by one step and no frac difference of one step is evidence
of anything on its own.

`last ep` is the highest epoch with a recorded val10 metric, which is NOT always the epoch the run
reached -- some runs evaluated every N epochs, and a few wrote metrics past the point their
`--epochs` suggests because they were resumed. `best ... @ep` is the best single epoch, which for a
run that decays afterwards is not what you would have shipped unless you were early-stopping.

Sorted by FINAL frac then FINAL DockQ, because the final epoch is the only column that does not
require a stopping rule to interpret.

| run | objective | trainable | L2-SP | last ep | best DockQ @ep | best frac @ep | final DockQ | final frac |
|---|---|---|---|---|---|---|---|---|
| `4y3ve6` | cross softplus (no PAE) | perblock-share | -- | 20 | 0.3108 @15 | 0.6111 @10 | **0.3089** | **0.6111** |
| `24u34d` | cross softplus (no PAE) | perblock-copy+orient | 3e-04 | 7 | 0.2893 @7 | 0.6111 @5 | **0.2893** | **0.6111** |
| `bjwcm5` | all softplus tau0.027 (no PAE) | perblock-copy+orient | 3e-04 | 15 | 0.2876 @5 | 0.6111 @7 | **0.2845** | **0.6111** |
| `m8rote` | PAE | full head 19.8M | -- | 3 | 0.3239 @3 | 0.5556 @3 | **0.3239** | **0.5556** |
| `ntwjfp` | cross softplus + PAE | perblock-share | -- | 18 | 0.3226 @11 | 0.5556 @2 | **0.3037** | **0.5556** |
| `hronsr` | all softplus L=8 tau0.027 (no PAE) | perblock-copy+orient | 3e-04 | 15 | 0.3042 @8 | 0.5556 @2 | **0.3024** | **0.5556** |
| `ykhvzs` | cross softplus + PAE | perblock-copy+orient | 3e-04 | 13 | 0.2914 @11 | 0.5556 @11 | **0.2914** | **0.5556** |
| `zfahpb` | cross softplus (no PAE) | perblock-copy | 3e-04 | 6 | 0.2877 @6 | 0.5556 @2 | **0.2877** | **0.5556** |
| `ggfhj5` | PAE | base | -- | 11 | 0.3099 @2 | 0.5556 @1 | **0.2820** | **0.5556** |
| `24u342` | cross softplus (no PAE) | perblock-copy+orient | 3e-04 | 50 | 0.3050 @28 | 0.6111 @9 | **0.2790** | **0.5556** |
| `ctij23` | cross expected tau0.1 (no PAE) | base | 1e-03 | 13 | 0.2988 @4 | 0.5556 @1 | **0.2780** | **0.5556** |
| `e4tvxe` | PAE | full head 19.8M | -- | 6 | 0.2753 @6 | 0.5556 @6 | **0.2753** | **0.5556** |
| `zfahp2` | cross softplus (no PAE) | perblock-copy | 3e-04 | 10 | 0.3071 @8 | 0.5556 @8 | **0.2743** | **0.5556** |
| `eewbr5` | PAE | full head 19.8M | -- | 30 | 0.3338 @27 | 0.5556 @4 | **0.3014** | **0.5000** |
| `h7msge` | cross softplus (no PAE) | base | 1e-03 | 42 | 0.3071 @2 | 0.5556 @1 | **0.2986** | **0.5000** |
| `8eiek8` | PAE | perblock-share | -- | 50 | 0.3246 @10 | 0.6111 @10 | **0.2913** | **0.5000** |
| `zbbci6` | cross softplus + PAE | perblock-share | -- | 16 | 0.2860 @4 | 0.5000 @1 | **0.2845** | **0.5000** |
| `xtq4vh` | PAE | perblock-share | -- | 1 | 0.2782 @1 | 0.5000 @1 | **0.2782** | **0.5000** |
| `aup7ku` | cross softplus + PAE | perblock-copy+orient | 5e-03 | 1 | 0.2749 @1 | 0.5000 @1 | **0.2749** | **0.5000** |
| `zcuiab` | PAE | base | -- | 21 | 0.2838 @20 | 0.5556 @12 | **0.2734** | **0.5000** |
| `oe6sru` | cross softplus + PAE | perblock-copy+orient | 5e-04 | 15 | 0.3029 @4 | 0.5556 @2 | **0.2712** | **0.5000** |
| `6v6bbf` | all softplus L=8 tau0.027 (no PAE) | perblock-copy+orient | 3e-04 | 15 | 0.2821 @1 | 0.5556 @2 | **0.2638** | **0.5000** |
| `mdv9x7` | PAE | full head 19.8M | -- | 30 | 0.2904 @23 | 0.6111 @13 | **0.2316** | **0.5000** |
| `xmz9ge` | all softplus L=8 tau0.027 (no PAE) | perblock-copy+orient | 5e-05 | 9 | 0.3034 @3 | 0.6111 @7 | **0.2306** | **0.5000** |
| `jjvdhk` | PAE | full head 19.8M | 1e-05 | 1 | 0.2687 @1 | 0.4444 @1 | **0.2687** | **0.4444** |
| `4x8t44` | PAE | perblock-share | 1e-02 | 50 | 0.2866 @15 | 0.5000 @1 | **0.2588** | **0.4444** |
| `2u8tf4` | PAE | full head 19.8M | -- | 133 | 0.2251 @133 | 0.4444 @133 | **0.2251** | **0.4444** |
| `nzqhdf` | cross softplus (no PAE) | perblock-copy+orient | 3e-04 | 6 | 0.2815 @3 | 0.5556 @3 | **0.1778** | **0.3889** |
| `335psh` | PAE | full head 19.8M | -- | 20 | 0.3045 @2 | 0.6111 @4 | **0.2098** | **0.3333** |
| `f8rqck` | PAE | full head 19.8M | -- | 10 | 0.2222 @8 | 0.3889 @8 | **0.2006** | **0.3333** |
| `wpg274` | PAE | full head 19.8M | 1e-05 | 50 | 0.2889 @16 | 0.6111 @12 | **0.1616** | **0.2778** |
| **SHIPPED** | *(untrained head)* | -- | -- | -- | 0.2909 | 0.5556 | **0.2909** | **0.5556** |
