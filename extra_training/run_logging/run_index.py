#!/usr/bin/env python
"""Index every run under training/runs into a searchable SQLite database.

    python -I run_logging/run_index.py --build          # (re)build training/runs/index.db
    python -I run_logging/run_index.py --find "pairs=within"
    python -I run_logging/run_index.py --show hronsr
    python -I run_logging/run_index.py --sql "SELECT rid, best_dockq FROM runs ORDER BY ..."

WHY THIS EXISTS. Answering "have we run a within-seed softplus arm?" meant grepping 94 run
directories by hand and reading args.json one at a time, and the answer was wrong the first time
it was asked: four runs DID train within-seed pairs, but with the linear form before --rank-form
existed, so "we have never run within-seed" and "we ran within-seed four times" were both true
depending on what you meant. A field-by-field table makes that distinction visible instead of
something to remember.

EVERY FIELD IS READ FROM THE RUN'S OWN ARTIFACTS, never from notes or from this session:
  args.json        the flags it was launched with -- the objective, trainable set, data, lambda
  status.json      exit code and wall time; absent means it never wrote one (killed or running)
  rank_rho_b*_v*   the held-out metrics, taking the HIGHEST cache version present, since the
                   version is bumped when what the numbers MEAN changes
  run.log          the trainable tensor count and parameter total the trainer actually resolved,
                   which is the only place --train-only's regex is expanded

THE CACHE VERSION IS RECORDED PER RUN. A v3 xsloss is the canonical cross-seed loss, a v4+ one is
the run's own objective; comparing them across that boundary is the mistake this column exists to
prevent.

NOTHING IS INFERRED. A field the artifacts do not carry is NULL, not a guess. In particular an
`epochs` of 50 with 9 metric rows means it was stopped, and both numbers are stored.
"""
import argparse
import csv
import glob
import json
import os
import re
import sqlite3
import sys

RUNS = os.path.join("training", "runs")
DB = os.path.join(RUNS, "index.db")

FLAT = [
    ("objective", ["rank_form", "rank_pairs", "rank_within_weight", "rank_topk_decay",
                   "rank_tau", "rank_cross_seed", "rank_weight_by_gap", "alpha_rank",
                   "alpha_pae", "alpha_confidence", "rank_structs_per_step"]),
    ("model", ["train_only", "pose_perblock", "pose_perblock_copy", "pose_perblock_tied",
               "pose_perblock_share", "frame_direction", "lora_rank", "blocks_per_ckpt"]),
    ("reg", ["l2_sp", "l2_sp_exclude", "lr", "warmup_steps", "clip_norm"]),
    ("data", ["manifest_selection", "val_loss_selection", "train_eval_pool", "seeds",
              "val_loss_seeds", "train_eval_budget", "epochs", "nsample", "accum", "crop_size"]),
]
COLS = [c for _, cs in FLAT for c in cs]


def _pool(path):
    if not path:
        return None
    return os.path.basename(os.path.dirname(os.path.abspath(path))) or None


def _metrics(d):
    """Highest-version rank_rho cache -> (version, n_epochs, last_row, best_dockq, best_ep)."""
    caches = sorted(glob.glob(os.path.join(d, "rank_rho_b*_v*.csv")))
    if not caches:
        return None, 0, {}, None, None
    cache = sorted(caches, key=lambda p: int(re.search(r"_v(\d+)\.csv$", p).group(1)))[-1]
    ver = int(re.search(r"_v(\d+)\.csv$", cache).group(1))
    rows = []
    with open(cache) as fh:
        rows = [r for r in csv.DictReader(fh)]
    if not rows:
        return ver, 0, {}, None, None
    rows.sort(key=lambda r: int(r["epoch"]))
    best_ep = best = None
    for r in rows:
        try:
            v = float(r["dockq_mean"])
        except (KeyError, ValueError, TypeError):
            continue
        if best is None or v > best:
            best, best_ep = v, int(r["epoch"])
    return ver, len(rows), rows[-1], best, best_ep


def _trainable(d):
    """(n_tensors, n_params) as the TRAINER resolved them -- the only place the regex expands."""
    p = os.path.join(d, "run.log")
    if not os.path.exists(p):
        return None, None
    pat = re.compile(r"trainable: (\d+) tensor\(s\), [\d.]+M of")
    pat2 = re.compile(r"--train-only .*?: (\d+) of \d+ head tensor\(s\), ([\d,]+) of")
    nt = npar = None
    with open(p, errors="replace") as fh:
        for line in fh:
            m = pat2.search(line)
            if m:
                nt, npar = int(m.group(1)), int(m.group(2).replace(",", ""))
                break
            m = pat.search(line)
            if m and nt is None:
                nt = int(m.group(1))
    return nt, npar


def build():
    con = sqlite3.connect(DB)
    con.execute("DROP TABLE IF EXISTS runs")
    con.execute(f"""CREATE TABLE runs (
        rid TEXT PRIMARY KEY, note TEXT, resume_of TEXT,
        exit_code INTEGER, wall_min REAL, ended TEXT, status TEXT,
        n_tensors INTEGER, n_params INTEGER,
        cache_version INTEGER, n_epochs INTEGER,
        train_pool TEXT, val_pool TEXT,
        last_dockq REAL, last_frac REAL, last_rho REAL, last_xsloss REAL,
        best_dockq REAL, best_dockq_epoch INTEGER,
        {', '.join(c + ' TEXT' for c in COLS)} )""")
    n = 0
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        rid = os.path.basename(d)
        if not (os.path.isdir(d) and re.fullmatch(r"[a-z0-9]{6}", rid) and rid != "latest"):
            continue
        try:
            a = json.load(open(os.path.join(d, "args.json")))
        except (OSError, ValueError):
            a = {}
        st = {}
        try:
            st = json.load(open(os.path.join(d, "status.json")))
        except (OSError, ValueError):
            pass
        ver, neps, last, best, best_ep = _metrics(d)
        nt, npar = _trainable(d)
        status = ("running" if not st else
                  ("ok" if st.get("exit_code") == 0 else f"exit {st.get('exit_code')}"))
        row = dict(rid=rid, note=(a.get("note") or "")[:4000],
                   resume_of=(os.path.basename(os.path.dirname(a["resume"]))
                              if a.get("resume") else None),
                   exit_code=st.get("exit_code"), wall_min=st.get("wall_min"),
                   ended=st.get("ended"), status=status,
                   n_tensors=nt, n_params=npar, cache_version=ver, n_epochs=neps,
                   train_pool=_pool(a.get("manifest_selection")),
                   val_pool=_pool(a.get("val_loss_selection")),
                   last_dockq=_f(last.get("dockq_mean")), last_frac=_f(last.get("frac_mean")),
                   last_rho=_f(last.get("rho_trained")), last_xsloss=_f(last.get("xsloss_trained")),
                   best_dockq=best, best_dockq_epoch=best_ep,
                   **{c: (None if a.get(c) is None else str(a.get(c))) for c in COLS})
        con.execute(f"INSERT INTO runs ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                    list(row.values()))
        n += 1
    con.commit()
    print(f"indexed {n} run(s) -> {DB}")
    return con


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--find", help="key=value filters, comma separated, e.g. "
                                   "'rank_pairs=within,train_pool=overfit30'")
    ap.add_argument("--show", help="one rid, every stored field")
    ap.add_argument("--sql", help="raw SELECT against table `runs`")
    ap.add_argument("--cols", default="rid,status,n_epochs,n_params,train_pool,val_pool,"
                                      "rank_form,rank_pairs,rank_within_weight,rank_topk_decay,"
                                      "rank_tau,alpha_pae,l2_sp,last_dockq,last_frac")
    a = ap.parse_args()
    if a.build or not os.path.exists(DB):
        con = build()
    else:
        con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    if a.show:
        r = con.execute("SELECT * FROM runs WHERE rid=?", (a.show,)).fetchone()
        if not r:
            raise SystemExit(f"no run {a.show!r} in the index")
        for k in r.keys():
            if k == "note":
                continue
            print(f"  {k:24s} {r[k]}")
        print(f"\n  note: {(r['note'] or '')[:1500]}")
        return
    if a.sql:
        for r in con.execute(a.sql):
            print(" | ".join(str(r[k]) for k in r.keys()))
        return
    where, args = "", []
    if a.find:
        parts = []
        for kv in a.find.split(","):
            k, _, v = kv.partition("=")
            parts.append(f"CAST({k.strip()} AS TEXT) = ?")
            args.append(v.strip())
        where = " WHERE " + " AND ".join(parts)
    cols = [c.strip() for c in a.cols.split(",")]
    rows = list(con.execute(f"SELECT {','.join(cols)} FROM runs{where} ORDER BY rid", args))
    if not rows:
        print("no matching run")
        return
    w = [max(len(c), max(len(str(r[c])) for r in rows)) for c in cols]
    print("  ".join(c.ljust(x) for c, x in zip(cols, w)))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print("  ".join(str(r[c]).ljust(x) for c, x in zip(cols, w)))
    print(f"\n{len(rows)} run(s)")


if __name__ == "__main__":
    main()
