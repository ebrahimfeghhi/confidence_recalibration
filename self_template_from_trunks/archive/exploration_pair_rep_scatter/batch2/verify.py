"""usage: verify.py A | C   -- exit 0 if complete; prints what is missing (as stage-specific lists)."""
import csv, glob, os, sys, collections, re
B = '/home/ebrahim/tts_abag/pair_rep_scatter/batch2'; PDBS = ['7yxu', '7zoz', '8be3', '8ux6']
FB = '/home/ebrahim/tts_abag/experimental_data/protenix/foldbench'
roles = [r for r in csv.DictReader(open(FB + '/chain_roles.csv')) if r['pdb'] in PDBS]
nif = collections.Counter(r['pdb'] for r in roles)
bad = 0
if sys.argv[1] == 'A':
    rows = collections.Counter()
    if os.path.exists(B + '/free/results.csv'):
        for r in csv.DictReader(open(B + '/free/results.csv')): rows[r['iface'].split('_')[0]] += 1
    for p in PDBS:
        nt = len(glob.glob(f'{B}/trunks/{p}/free_*.npz')); exp = 250 * nif[p]
        print(f'  {p}: trunks {nt}/50, scored rows {rows[p]}/{exp}'); bad += (nt != 50) + (rows[p] != exp)
else:
    donors = {p: sum(1 for l in open(f'{B}/select/queue_{p}.tsv') if l.strip()) for p in PDBS}
    for c in (1, 3, 5, 10):
        for kind in ('msa', 'row1'):
            arm = f'{kind}_c{c}'; rows = collections.Counter()
            f = f'{B}/resume/{arm}/results.csv'
            if os.path.exists(f):
                for r in csv.DictReader(open(f)): rows[r['iface'].split('_')[0]] += 1
            miss = {p: donors[p] * 5 * nif[p] - rows[p] for p in PDBS if rows[p] != donors[p] * 5 * nif[p]}
            print(f'  {arm}: ' + ('complete' if not miss else f'MISSING rows {miss}')); bad += bool(miss)
sys.exit(1 if bad else 0)
