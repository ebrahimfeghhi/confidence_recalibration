"""`trainer/context.py` -- the state the extracted modules used to read from `main()`'s scope.

WHY THIS EXISTS. `unit.py` and `loop.py` were closures inside an 8,402-line function: they read
75 enclosing locals directly, which is exactly what made that function impossible to split. Each
of those reads is now `ctx.<name>`, so the dependencies are declared rather than implied, and a
missing one is an AttributeError at the call site instead of a NameError 900 steps in.

It is a plain mutable namespace on purpose. The review on issue #9 asked for an immutable
RunConfig plus a mutable StepState; splitting these 75 names into those two is worth doing, but
doing it in the same change as the extraction would mean two unreviewable diffs at once. This is
the first step: make the dependencies explicit. The second is to classify them.
"""


class Ctx:
    """Everything the extracted modules read. Attributes are set by `main()` as it builds them."""

    #: every name the extraction found, so a typo is a construction-time failure, not a runtime one
    SLOTS = (
        'BC',
        'DOPS',
        'PFC',
        'ProtenixLoss',
        'RKL_probe',
        'TERMS',
        '_CROP_TOK_DIGEST',
        '_EXTRA_EVAL',
        '_LRS',
        '_LR_STEPS',
        '_T',
        '_TN',
        '_TOK_DROP',
        '_Zero',
        '__file__',
        '_ckpt_hist',
        '_crop_epoch',
        '_dead_head',
        '_dops',
        '_drift_history',
        '_dropped_terms',
        '_dt',
        '_git_commit',
        '_iface_of',
        '_iptm_fh',
        '_l2sp_drift',
        '_lr_horizon',
        '_manifest_digest',
        '_mem_ep',
        '_pae_unw',
        '_pf_started',
        '_pf_stat',
        '_pfc',
        '_probe',
        '_rank_div',
        '_rank_lab',
        '_rank_seen',
        '_rel_stat',
        '_scored_asym',
        '_step_poses',
        '_step_poses_g',
        '_tok_drop_stat',
        '_tr_on',
        '_tr_stat',
        '_train_eval_budget',
        '_train_eval_seed_base',
        '_xs_coef',
        '_xs_div',
        '_xs_stat',
        'a',
        'accum_eff',
        'accum_local',
        'build_epoch_cross_seed',
        'configs',
        'crop_cols',
        'crop_p',
        'crop_rows',
        'crop_stats',
        'dev',
        'eval_checkpoint',
        'head_param_names',
        'is_main',
        'loss_fn',
        'model',
        'opt',
        'prefixes',
        'rank',
        'run_unit',
        'trainable',
        'trainable_params',
        'traj_cols',
        'traj_p',
        'traj_rank_p',
        'val_loss_checkpoint',
        'world',
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def require(self, *names):
        """Fail loudly and early if something downstream will need a name nobody set."""
        missing = [n for n in names if not hasattr(self, n)]
        if missing:
            raise RuntimeError(
                f"STOP: trainer context is missing {len(missing)} name(s) the next stage reads: "
                f"{missing[:8]}. This is a wiring bug in main(), not a config error.")
