import csv, os, collections, sys, re, glob
P = '/home/ebrahim/tts_abag/pair_rep_scatter'; FB = '/home/ebrahim/tts_abag/experimental_data/protenix/foldbench'
nif = collections.Counter(r['pdb'] for r in csv.DictReader(open(FB + '/chain_roles.csv')))
bad = 0; on = 0; off = 0
for d, p in [('batch2', x) for x in ('7yxu', '7zoz', '8be3', '8ux6')] + [('trunk_pilot', x) for x in ('9fww', '8fxv')]:
    donors = sum(1 for l in open(f'{P}/{d}/select/queue_{p}_ownseed.tsv') if l.strip()); exp = donors * 5 * nif[p]
    f = f'{P}/{d}/resume/msa_c3_ownseed_nodrop/results.csv'; n = sum(1 for r in csv.DictReader(open(f)) if r['iface'].split('_')[0] == p) if os.path.exists(f) else 0
    print(f'  {d}/{p}: rows {n}/{exp}'); bad += (n != exp)
    for lf in glob.glob(f'{P}/{d}/resume/msa_c3_ownseed_nodrop/w*_{p}.log'):
        for l in open(lf, errors='ignore'):
            m = re.search(r'MC_DROPOUT decision=(True|False)', l)
            if m: on += m.group(1) == 'True'; off += m.group(1) == 'False'
print(f'  dropout decisions logged: on={on} off={off} (expected on=0)'); bad += (on != 0)
sys.exit(1 if bad else 0)
