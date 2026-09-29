import csv, os, collections, sys
P = '/home/ebrahim/tts_abag/pair_rep_scatter'; FB = '/home/ebrahim/tts_abag/experimental_data/protenix/foldbench'
nif = collections.Counter(r['pdb'] for r in csv.DictReader(open(FB + '/chain_roles.csv')))
jobs = [('batch2', p) for p in ('7yxu', '7zoz', '8be3', '8ux6')] + [('trunk_pilot', p) for p in ('9fww', '8fxv')]
bad = 0
for d, p in jobs:
    donors = sum(1 for l in open(f'{P}/{d}/select/queue_{p}_ownseed.tsv') if l.strip()); exp = donors * 5 * nif[p]; n = 0
    f = f'{P}/{d}/resume/msa_c3_ownseed/results.csv'
    if os.path.exists(f): n = sum(1 for r in csv.DictReader(open(f)) if r['iface'].split('_')[0] == p)
    print(f'  {d}/{p}: rows {n}/{exp}'); bad += (n != exp)
f = f'{P}/trunk_pilot/resume/_logmc_control_fixedseed/results.csv'; n = sum(1 for _ in csv.DictReader(open(f))) if os.path.exists(f) else 0
print(f'  control (fixed seed, 1 donor): rows {n}/5'); bad += (n != 5)
sys.exit(1 if bad else 0)
