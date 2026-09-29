"""Shadow FoldBench dataset root for batch2 (7yxu, 7zoz, 8be3, 8ux6): same construction as trunk_pilot/build_shadow.py, generalized.
free/<pdb>/<seed>-legacy-no-provenance -> the new run dir; chain_maps.csv from each pose's own OST json(s) (roles only; union over interface
jsons, asserted consistent); chain_roles.csv = the benchmark's roles; pool.csv from this run's scored results."""
import csv, glob, json, os, re
B = '/home/ebrahim/tts_abag/pair_rep_scatter/batch2'; SH = B + '/dataset'
FB = '/home/ebrahim/tts_abag/experimental_data/protenix/foldbench'
PDBS = ['7yxu', '7zoz', '8be3', '8ux6']
def first_copy(cm):
    out = dict(cm)
    for k in list(cm):
        m = re.fullmatch(r"(.+)-(\d+)", k)
        if m and m.group(1) not in cm: out[m.group(1)] = out.pop(k)
    return out
roles = [r for r in csv.DictReader(open(FB + '/chain_roles.csv')) if r['pdb'] in PDBS]
assert {r['pdb'] for r in roles} == set(PDBS), sorted({r['pdb'] for r in roles})
with open(SH + '/chain_roles.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(roles[0].keys())); w.writeheader(); w.writerows(roles)
need = {p: {c for r in roles if r['pdb'] == p for c in (r['ab_chain'], r['ag_chain'])} for p in PDBS}
maps, nposes = [], {}
for pdb in PDBS:
    seeds = {}
    for rd in sorted(glob.glob(f'{B}/free/{pdb}/*/')):
        seed = int(os.path.basename(rd.rstrip('/')).split('-')[0]); assert seed not in seeds, (pdb, seed); seeds[seed] = rd.rstrip('/')
    assert sorted(seeds) == list(range(101, 151)), (pdb, len(seeds))
    os.makedirs(f'{SH}/free/{pdb}', exist_ok=True)
    for seed, rd in seeds.items():
        link = f'{SH}/free/{pdb}/{seed}-legacy-no-provenance'
        if os.path.lexists(link): os.remove(link)
        os.symlink(rd, link)
        for s in range(5):
            assert os.path.exists(f'{rd}/predictions/{pdb}_sample_{s}.cif'), (rd, s)
            fs = sorted(glob.glob(f'{rd}/ost_json/**/{s}.json', recursive=True)); assert fs, (rd, s)
            m = {}
            for f in fs:
                for k, v in first_copy(json.load(open(f))['chain_mapping']).items():
                    assert m.get(k, v) == v, ('conflicting chain map', rd, s, k); m[k] = v
            assert need[pdb] <= set(m), ('roles not covered by chain map', pdb, seed, s, sorted(need[pdb]), m)
            maps.append((pdb, seed, s, json.dumps(m)))
    nposes[pdb] = len(seeds) * 5
with open(SH + '/chain_maps.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['pdb', 'seed', 'sample', 'chain_mapping']); w.writerows(maps)
ci = {r['interface']: (r['chain_1'], r['chain_2']) for r in roles}; rows = 0; keys = set()
with open(SH + '/pool.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['pdb', 'chain1', 'chain2', 'seed', 'sample', 'dockq', 'pair_iptm'])
    for r in csv.DictReader(open(B + '/free/results.csv')):
        if r['iface'] in ci:
            k = (r['iface'], r['seed'], r['sample']); assert k not in keys, ('duplicate result row', k); keys.add(k)
            w.writerow([r['iface'].split('_')[0], ci[r['iface']][0], ci[r['iface']][1], r['seed'], r['sample'], r['dockq'], r['iptm']]); rows += 1
expect = sum(nposes[r['pdb']] for r in roles)
print('poses per pdb', nposes, '| interfaces', len(roles), '| chain_maps rows', len(maps), '| pool.csv rows', rows, '(expected', expect, ')')
assert rows == expect, (rows, expect)
