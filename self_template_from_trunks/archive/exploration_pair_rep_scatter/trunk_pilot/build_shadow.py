"""Shadow FoldBench dataset root for the trunk pilot: the pipeline's inputs for 9fww and 8fxv, pointing at the NEW free folds.
free/<pdb>/<seed>-legacy-no-provenance -> the new run dir; chain_maps.csv from each pose's own OST json (roles only);
chain_roles.csv = the benchmark's roles (known input); pool.csv from this run's scored results."""
import csv, glob, json, os, sys
D = '/home/ebrahim/tts_abag/pair_rep_scatter/trunk_pilot'; SH = D + '/dataset'
FB = '/home/ebrahim/tts_abag/experimental_data/protenix/foldbench'

import re
def first_copy(cm):
    """Same rule as foldbench_setup: a pose mapped onto reference copy N ("A-2") is recorded under the first copy's
    names ("A") when the first copy is unmapped. The map is read for roles only."""
    out = dict(cm)
    for k in list(cm):
        m = re.fullmatch(r"(.+)-(\d+)", k)
        if m and m.group(1) not in cm:
            out[m.group(1)] = out.pop(k)
    return out
PDBS = ['9fww', '8fxv']
roles = [r for r in csv.DictReader(open(FB + '/chain_roles.csv')) if r['pdb'] in PDBS]
assert {r['pdb'] for r in roles} == set(PDBS), roles
with open(SH + '/chain_roles.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(roles[0].keys())); w.writeheader(); w.writerows(roles)
maps, nposes = [], {}
for pdb in PDBS:
    runs = sorted(glob.glob(f'{D}/free/{pdb}/*/'))
    seeds = {}
    for rd in runs:
        seed = int(os.path.basename(rd.rstrip('/')).split('-')[0]); assert seed not in seeds, (pdb, seed)
        seeds[seed] = rd.rstrip('/')
    assert sorted(seeds) == list(range(101, 151)), (pdb, sorted(seeds)[:3], len(seeds))
    os.makedirs(f'{SH}/free/{pdb}', exist_ok=True)
    for seed, rd in seeds.items():
        link = f'{SH}/free/{pdb}/{seed}-legacy-no-provenance'
        if os.path.lexists(link): os.remove(link)
        os.symlink(rd, link)
        for s in range(5):
            assert os.path.exists(f'{rd}/predictions/{pdb}_sample_{s}.cif'), (rd, s)
            j = json.load(open(f'{rd}/ost_json/{s}.json')); maps.append((pdb, seed, s, json.dumps(first_copy(j['chain_mapping']))))
    nposes[pdb] = len(seeds) * 5
with open(SH + '/chain_maps.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['pdb', 'seed', 'sample', 'chain_mapping']); w.writerows(maps)
rows = 0
with open(SH + '/pool.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['pdb', 'chain1', 'chain2', 'seed', 'sample', 'dockq', 'pair_iptm'])
    ci = {r['interface']: (r['chain_1'], r['chain_2']) for r in roles}
    for r in csv.DictReader(open(D + '/free/results.csv')):
        if r['iface'] in ci:
            w.writerow([r['iface'].split('_')[0], ci[r['iface']][0], ci[r['iface']][1], r['seed'], r['sample'], r['dockq'], r['iptm']]); rows += 1
print('poses per pdb', nposes, '| chain_maps rows', len(maps), '| pool.csv rows', rows)
assert rows == sum(nposes.values()), (rows, nposes)
