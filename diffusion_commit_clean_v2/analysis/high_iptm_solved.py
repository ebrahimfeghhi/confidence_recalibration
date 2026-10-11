#!/usr/bin/env python
"""How many interfaces whose top free-50 pair ipTM exceeds a threshold are solved (top-ipTM pose DockQ >= 0.23).
Uses score_experiment's pool_by_seed / classes_at, so classes match every other report. Written 2026-10-02.
Usage: python analysis/high_iptm_solved.py [threshold]   (default 0.9)"""
import collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import score_experiment as se, paths
T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
for name in ('foldbench', 'pxm2025'):
    pool, first = se.pool_by_seed(50, str(paths.DATASETS[name].pool_csv))
    cls = se.classes_at(pool, first)
    top = {}
    for i in cls:
        keep = [r for r in pool[i] if r[0] in first[i.split('_')[0]]]
        top[i] = max(keep, key=lambda r: r[3])
        assert (top[i][2] >= se.ACCEPTABLE) == (cls[i] == 'solved')
    hi = [i for i in cls if top[i][3] > T]
    print(f"{name}: interfaces {len(cls)} classes {dict(collections.Counter(cls.values()))}")
    print(f"  top ipTM > {T}: {len(hi)} interfaces -> {dict(collections.Counter(cls[i] for i in hi))}")
    for i in sorted(hi):
        if cls[i] != 'solved':
            print(f"    not solved: {i} {cls[i]} top ipTM {top[i][3]:.3f} DockQ {top[i][2]:.3f}")
