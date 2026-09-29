#!/usr/bin/env python
"""Pipelined FoldBench analysis: per complex  A (50 free folds + bf16 trunks) -> B (CPU donor selection + templates)
-> C (two arms from the saved trunks, MSA on, 3 recycles, MC dropout OFF: T = each donor refolded with its pose as template, from its own seed's
trunk; K = compute-matched control, the same number of folds resumed from trunks with NO template) -> verify -> summarize -> delete trunks.

One controller keeps 4 GPU slots busy across complexes. State is DERIVED FROM ARTIFACTS ON DISK, so a restart resumes:
  todo    no data dir
  A       free folds / trunks incomplete (verify_A false)
  B       verify_A true, select/B_OK missing        (CPU subprocess, max --max-b at once)
  C       select/B_OK present, verify_C false
  D       verify_C true: summarize, then delete the trunks
  done    DONE.json present          parked  PARKED.txt present (reason inside)

USAGE  pipeline.py run --gpus 0,1,2,5 [--cap-gb 450] [--only a,b] [--max-b 2]     controller (run in tmux)
       pipeline.py stageB <pdb>                                                    one complex's CPU stage (used internally)
       pipeline.py status                                                          table of every complex's state
STOP:  touch pipeline/STOP  -> no new jobs; exits when the running ones finish.
RULES: trunks are deleted ONLY after verify_C passed AND the summary row was written AND one trunk (donor A00's seed) was copied to kept_trunks/ and verified; every deletion is logged in deletions.log.
"""
import argparse, collections, csv, glob, json, os, re, shutil, statistics, subprocess, sys, time

EXP = '/home/ebrahim/tts_abag/self_template_from_trunks'; ROOT = EXP; DATA = EXP + '/data'
H = '/home/ebrahim/tts_abag/diffusion_commit_clean_v2'; PY = '/home/ebrahim/Protenix/.venv/bin/python'
FB = '/home/ebrahim/tts_abag/experimental_data/protenix/foldbench'; JD = '/home/ebrahim/Protenix/foldbench_abag/jsons'
DONE_ALREADY = set()   # all 104 complexes are run in this experiment, including the 6 done earlier (rerun with the control arm)
ARM = 'msa_c3_ownseed_nodrop'; CTL = 'ctl_c3_nodrop'; NCYCLE_C = 3; SEEDS = list(range(101, 151)); QUOTA_LIMIT_GB = 1024
LOG = ROOT + '/controller.log'


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'; print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')


def load_info():
    roles = collections.defaultdict(list); hdr = None
    for r in csv.DictReader(open(FB + '/chain_roles.csv')): roles[r['pdb']].append(r)
    pool = {r['pdb'] for r in csv.DictReader(open(FB + '/pool.csv'))}; tok = {}
    for p in roles:
        f = f'{JD}/{p}-final-updated.json'
        if p not in pool or not os.path.exists(f): continue
        j = json.load(open(f)); j = j[0] if isinstance(j, list) else j; t = 0
        for s in j['sequences']:
            e = (s.get('proteinChain') or s.get('protein') or s.get('dnaSequence') or s.get('rnaSequence') or {}); t += e.get('count', 1) * len(e.get('sequence', ''))
        tok[p] = t
    return roles, tok


def R(p): return f'{DATA}/{p}'
def trunk_gb(n): return (n * n * 512 + n * 768) * len(SEEDS) / 1e9


def read_rows(path): return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def rows_ok(rows, expected):
    keys = collections.Counter((r['iface'], r['seed'], r['arm'], r['donor_pose'], r['sample']) for r in rows)
    dup = sum(v > 1 for v in keys.values())
    bad = sum(not (0 <= float(r['iptm']) <= 1 and 0 <= float(r['dockq']) <= 1) for r in rows)
    return len(rows) == expected and dup == 0 and bad == 0


def verify_A(p, roles):
    nt = len(glob.glob(f'{R(p)}/trunks/{p}/free_*.npz'))
    return nt == len(SEEDS) and rows_ok(read_rows(f'{R(p)}/free/results.csv'), 5 * len(SEEDS) * len(roles[p]))


def n_donors(p):
    q = f'{R(p)}/select/queue_ownseed.tsv'
    return sum(1 for l in open(q) if l.strip()) if os.path.exists(q) else 0


def verify_C(p, roles):
    d = n_donors(p)
    return d > 0 and rows_ok(read_rows(f'{R(p)}/resume/{ARM}/results.csv'), 5 * d * len(roles[p]))


def n_ctl(p):
    q = f'{R(p)}/select/queue_ctl.tsv'
    return sum(1 for l in open(q) if l.strip()) if os.path.exists(q) else 0


def verify_K(p, roles):
    d = n_ctl(p)
    return d > 0 and rows_ok(read_rows(f'{R(p)}/resume/{CTL}/results.csv'), 5 * d * len(roles[p]))


def status(p, roles):
    r = R(p)
    if os.path.exists(f'{r}/PARKED.txt'): return 'parked'
    if os.path.exists(f'{r}/DONE.json'): return 'done'
    if not os.path.isdir(r): return 'todo'
    if not verify_A(p, roles): return 'A'
    if not os.path.exists(f'{r}/select/B_OK'): return 'B'
    if not (verify_C(p, roles) and verify_K(p, roles)): return 'C'
    return 'D'


def park(p, why):
    os.makedirs(R(p), exist_ok=True); open(f'{R(p)}/PARKED.txt', 'w').write(why + '\n'); log(f'PARKED {p}: {why}')


def joinable(p, stage, kind='T'):
    r = R(p)
    if stage == 'A': total, cd = len(SEEDS), f'{r}/free/claims'
    elif kind == 'T': total, cd = n_donors(p), f'{r}/resume/{ARM}/claims'
    else: total, cd = n_ctl(p), f'{r}/resume/{CTL}/claims'
    return total - (len(os.listdir(cd)) if os.path.isdir(cd) else 0)


def gpu_cmd(stage, p, gpu, kind='T'):
    r = R(p)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), LAYERNORM_TYPE='openfold', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    if stage == 'A':
        env.update(TRUNK_DUMP_DIR=f'{r}/trunks', PROTENIX_DUMP_TRUNK_BF16='1')
        cmd = [PY, 'folder.py', '--queue', f'{r}/queue_free.tsv', '--out', f'{r}/free', '--csv', f'{r}/free/results.csv', '--claim-dir', f'{r}/free/claims',
               '--dataset', 'foldbench', '--ncycle', '10', '--nsample', '5', '--dtype', 'bf16']
    elif kind == 'K':
        # compute-matched control: resume each seed's own trunk for the same 3 recycles with NO template (dropout off, MSA on)
        env.update(TRUNK_LOAD_DIR=f'{r}/trunks', PROTENIX_LOG_MC='1', PROTENIX_MC_DROPOUT_APPLY_RATE='0'); arm = f'{r}/resume/{CTL}'
        cmd = [PY, 'folder.py', '--queue', f'{r}/select/queue_ctl.tsv', '--out', arm, '--csv', f'{arm}/results.csv', '--claim-dir', f'{arm}/claims',
               '--dataset', 'foldbench', '--ncycle', str(NCYCLE_C), '--nsample', '5', '--dtype', 'bf16']
    else:
        env.update(TRUNK_LOAD_DIR=f'{r}/trunks', PROTENIX_LOG_MC='1', PROTENIX_MC_DROPOUT_APPLY_RATE='0'); arm = f'{r}/resume/{ARM}'   # dropout OFF in stage C (rate 0 still consumes the same draw)
        cmd = [PY, 'folder.py', '--queue', f'{r}/select/queue_ownseed.tsv', '--out', arm, '--csv', f'{arm}/results.csv', '--claim-dir', f'{arm}/claims',
               '--dataset', 'foldbench', '--templates', f'{r}/select/templates', '--ncycle', str(NCYCLE_C), '--nsample', '5', '--dtype', 'bf16']
    return cmd, env


# ---------------------------------------------------------------- stage B (CPU)
def first_copy(cm):
    out = dict(cm)
    for k in list(cm):
        m = re.fullmatch(r'(.+)-(\d+)', k)
        if m and m.group(1) not in cm: out[m.group(1)] = out.pop(k)
    return out


def build_shadow(p, roles):
    r = R(p); sh = f'{r}/dataset'; os.makedirs(f'{sh}/free/{p}', exist_ok=True)
    rr = roles[p]
    with open(sh + '/chain_roles.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0].keys())); w.writeheader(); w.writerows(rr)
    need = {c for x in rr for c in (x['ab_chain'], x['ag_chain'])}; seeds = {}
    for rd in sorted(glob.glob(f'{r}/free/{p}/*/')):
        seed = int(os.path.basename(rd.rstrip('/')).split('-')[0]); assert seed not in seeds, ('two run dirs for seed', p, seed); seeds[seed] = rd.rstrip('/')
    assert sorted(seeds) == SEEDS, (p, len(seeds)); maps = []
    for seed, rd in seeds.items():
        link = f'{sh}/free/{p}/{seed}-legacy-no-provenance'
        if os.path.lexists(link): os.remove(link)
        os.symlink(rd, link)
        for s in range(5):
            assert os.path.exists(f'{rd}/predictions/{p}_sample_{s}.cif'), (rd, s)
            fs = sorted(glob.glob(f'{rd}/ost_json/**/{s}.json', recursive=True)); assert fs, ('no ost json', rd, s); m = {}
            for f in fs:
                for k, v in first_copy(json.load(open(f))['chain_mapping']).items():
                    assert m.get(k, v) == v, ('conflicting chain map', rd, s, k); m[k] = v
            assert need <= set(m), ('roles not covered by chain map', p, seed, s, sorted(need), m)
            maps.append((p, seed, s, json.dumps(m)))
    with open(sh + '/chain_maps.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['pdb', 'seed', 'sample', 'chain_mapping']); w.writerows(maps)
    ci = {x['interface']: (x['chain_1'], x['chain_2']) for x in rr}; n = 0; keys = set()
    with open(sh + '/pool.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['pdb', 'chain1', 'chain2', 'seed', 'sample', 'dockq', 'pair_iptm'])
        for x in read_rows(f'{r}/free/results.csv'):
            if x['iface'] in ci:
                k = (x['iface'], x['seed'], x['sample']); assert k not in keys, ('duplicate result row', k); keys.add(k)
                w.writerow([p, ci[x['iface']][0], ci[x['iface']][1], x['seed'], x['sample'], x['dockq'], x['iptm']]); n += 1
    assert n == 5 * len(SEEDS) * len(rr), (n, len(rr))


def stageB(p, roles):
    r = R(p); env = dict(os.environ, FOLDBENCH_DATA=f'{r}/dataset', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    build_shadow(p, roles)
    sel = f'{r}/select'; fr = f'{sel}/chain_frame'
    def run(cmd, e=env): subprocess.run(cmd, env=e, check=True)
    run([PY, f'{H}/build_experiment.py', '--dataset', 'foldbench', '--out', sel, '--stop-after', 'interfaces', '--jobs', '4'])
    run([PY, f'{H}/pose_to_pose_dockq.py', '--poses', f'{sel}/poses.csv', '--maps', f'{fr}/chain_maps_anchor.csv', '--roles', f'{sel}/interfaces.csv',
         '--out', f'{sel}/pose_dockq_matrices', '--device', 'cpu', '--dataset', 'foldbench'], dict(env, CUDA_VISIBLE_DEVICES=''))
    run([PY, f'{H}/structural_diversity_filter.py', '--matrices', f'{sel}/pose_dockq_matrices', '--out', f'{sel}/queue.tsv', '--cap', '50', '--seed', '430000', '--dataset', 'foldbench'])
    run([PY, f'{H}/build_templates.py', '--queue', f'{sel}/queue.tsv', '--ns', f'{sel}/templates', '--maps', f'{fr}/chain_maps_anchor.csv', '--roles', f'{sel}/interfaces.csv', '--jobs', '4', '--dataset', 'foldbench'])
    run([PY, f'{H}/preflight.py', '--ns', f'{sel}/templates', '--queue', f'{sel}/queue.tsv', '--maps', f'{fr}/chain_maps_anchor.csv', '--roles', f'{sel}/interfaces.csv'])
    out = []
    for l in open(f'{sel}/queue.tsv'):
        if not l.strip() or l.startswith('#'): continue
        f = l.rstrip('\n').split('\t'); m = re.search(r'/(\d+)-[^/]+/predictions/', f[5]); assert m, l; f[4] = m.group(1); out.append('\t'.join(f))
    assert out, 'no donors selected'
    open(f'{sel}/queue_ownseed.tsv', 'w').write('\n'.join(out) + '\n')
    # compute-matched control: the donors' distinct seeds first, then fill with other seeds to the SAME fold count as the templated arm
    seeds = []
    for l in out:
        sd = int(l.split('\t')[4])
        if sd not in seeds: seeds.append(sd)
    for sd in SEEDS:
        if len(seeds) >= min(len(out), len(SEEDS)): break
        if sd not in seeds: seeds.append(sd)
    open(f'{sel}/queue_ctl.tsv', 'w').write(''.join(f'free\t{p}\t*\tctl\t{sd}\n' for sd in seeds))
    open(f'{sel}/B_OK', 'w').write(f'{len(out)} donors, {len(seeds)} control folds\n')


# ---------------------------------------------------------------- health gate (must pass before ANY trunk is deleted)
def health(p, roles):
    """Returns (problems, metrics). Empty problems = healthy. Checks stage A dumps and BOTH stage-C arms (T templated, K compute-matched control)."""
    r = R(p); probs = []; met = {}
    la = f'{r}/gpu_A.log'; zl = [l for l in open(la, errors='ignore') if 'DUMP_TRUNK_BF16 z' in l] if os.path.exists(la) else []
    nd = [int(re.search(r'n_diff=(\d+)/', l).group(1)) for l in zl if re.search(r'n_diff=(\d+)/', l)]
    if len(nd) < len(SEEDS) or any(nd): probs.append(f'stage A bf16 dumps: {len(nd)} logged (need {len(SEEDS)}), {sum(1 for x in nd if x)} not exact')
    free_iptm = collections.defaultdict(list)
    for x in read_rows(f'{r}/free/results.csv'): free_iptm[(x['iface'], x['seed'])].append(float(x['iptm']))
    for kind, arm, logn, nf in (('T', ARM, 'gpu_CT.log', n_donors(p)), ('K', CTL, 'gpu_CK.log', n_ctl(p))):
        lf = f'{r}/{logn}'; txt = open(lf, errors='ignore').read() if os.path.exists(lf) else ''
        ncy = [int(x) for x in re.findall(r'PROTENIX_LOAD_TRUNK: recycling carry-over initialised[^\n]*?N_cycle=(\d+)', txt)]; rng = txt.count('RNG streams restored')
        off = len(re.findall(r'MC_DROPOUT decision=False', txt)); on = len(re.findall(r'MC_DROPOUT decision=True', txt))
        # what the user asked to have verified: every stage-C fold resumed a saved trunk for exactly NCYCLE_C recycles, with dropout off
        if len(ncy) < nf or any(c != NCYCLE_C for c in ncy) or rng < nf or off < nf or on:
            probs.append(f'arm {kind}: folds {nf}, trunk loads logged {len(ncy)} (N_cycle values {sorted(set(ncy))}, need {NCYCLE_C}), rng restores {rng}, dropout off {off} on {on}')
        rj = [json.load(open(f)) for f in glob.glob(f'{r}/resume/{arm}/{p}/*/run.json')]
        want_t = (kind == 'T')
        if len(rj) < nf or any(j.get('n_cycle') != NCYCLE_C or j.get('n_sample') != 5 or j.get('precision') != 'bf16' or bool(j.get('templated')) != want_t for j in rj):
            probs.append(f'arm {kind}: run.json check failed ({len(rj)} of {nf} folds; n_cycle {sorted({j.get("n_cycle") for j in rj})}, n_sample {sorted({j.get("n_sample") for j in rj})}, templated {sorted({bool(j.get("templated")) for j in rj})})')
        by = collections.defaultdict(list)
        for x in read_rows(f'{r}/resume/{arm}/results.csv'): by[(x['iface'], x['seed'], x['arm'])].append(float(x['iptm']))
        deltas = [abs(statistics.mean(v) - statistics.mean(free_iptm[(i, sd)])) for (i, sd, tag), v in by.items() if free_iptm.get((i, sd))]
        allv = [x for v in by.values() for x in v]
        # descriptive numbers only (they fill two summary columns); they are NOT checked against any threshold
        m = dict(med_abs_diptm_vs_free=round(statistics.median(deltas), 4) if deltas else 9.0, share_changed=round(sum(d > 1e-3 for d in deltas) / max(1, len(deltas)), 3),
                 med_iptm=round(statistics.median(allv), 4) if allv else 0.0, folds=nf,
                 verified=dict(trunk_loads=len(ncy), n_cycle=sorted(set(ncy)), rng_restores=rng, dropout_off=off, dropout_on=on, run_json=len(rj))); met[kind] = m
    return probs, met


# ---------------------------------------------------------------- summarize / delete
def summarize(p, roles, met):
    r = R(p); fr = collections.defaultdict(list); free_dq = {}
    for x in read_rows(f'{r}/free/results.csv'): fr[x['iface']].append((float(x['iptm']), float(x['dockq']))); free_dq[(x['iface'], int(x['seed']), int(x['sample']))] = float(x['dockq'])
    arm = collections.defaultdict(list)
    for x in read_rows(f'{r}/resume/{ARM}/results.csv'): arm[x['iface']].append((float(x['iptm']), float(x['dockq']), x['arm']))
    donors = []
    for l in open(f'{r}/select/queue_ownseed.tsv'):
        f = l.rstrip('\n').split('\t'); m = re.search(r'/(\d+)-[^/]+/predictions/\w+_sample_(\d+)\.cif', f[5]); donors.append((int(m.group(1)), int(m.group(2))))
    ctl = collections.defaultdict(list)
    for x in read_rows(f'{r}/resume/{CTL}/results.csv'): ctl[x['iface']].append((float(x['iptm']), float(x['dockq']), x['seed']))
    drop = collections.Counter()
    for lf in glob.glob(f'{r}/gpu_CT.log') + glob.glob(f'{r}/gpu_CK.log'):
        for l in open(lf, errors='ignore'):
            m = re.search(r'MC_DROPOUT decision=(True|False)', l)
            if m: drop[m.group(1)] += 1
    n_folds = n_donors(p) + n_ctl(p)
    # a decision must be logged for EVERY stage-C fold (donors + control) and none may be ON; otherwise the caller must not delete the trunks
    assert drop['True'] == 0 and drop['False'] >= n_folds, ('dropout check failed', dict(drop), 'expected >=', n_folds, 'folds, none on')
    rows = []
    for rr in roles[p]:
        i = rr['interface']; fp = max(fr[i]); ap = max(arm[i]); cp = max(ctl[i]); dv = [free_dq[(i, s, x)] for s, x in donors]
        rows.append(dict(pdb=p, interface=i, donors=len(donors), donors_acceptable=sum(d >= .23 for d in dv), best_donor_dockq=round(max(dv), 4),
                         free_pick_dockq=round(fp[1], 4), free_pick_iptm=round(fp[0], 4), free_acceptable=int(fp[1] >= .23),
                         arm_pick_dockq=round(ap[1], 4), arm_pick_iptm=round(ap[0], 4), arm_pick_donor=ap[2], arm_acceptable=int(ap[1] >= .23),
                         arm_share_poses_acceptable=round(sum(x[1] >= .23 for x in arm[i]) / len(arm[i]), 4),
                         ctl_folds=len({x[2] for x in ctl[i]}), ctl_pick_dockq=round(cp[1], 4), ctl_pick_iptm=round(cp[0], 4), ctl_acceptable=int(cp[1] >= .23),
                         ctl_share_poses_acceptable=round(sum(x[1] >= .23 for x in ctl[i]) / len(ctl[i]), 4), dropout_true=drop['True'], dropout_false=drop['False'], n_stageC_folds=n_folds,
                         ctl_med_abs_diptm_vs_free=met['K']['med_abs_diptm_vs_free'], arm_med_abs_diptm_vs_free=met['T']['med_abs_diptm_vs_free']))
    sc = ROOT + '/summary_all.csv'; new = not os.path.exists(sc)
    with open(sc, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new: w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(f'{r}/summary.json', 'w'), indent=1)
    return rows


def delete_trunks(p):
    t = f'{R(p)}/trunks'
    assert os.path.isdir(t) and not os.path.islink(t) and os.path.realpath(t).startswith(DATA + '/'), t
    kb = int(subprocess.run(['du', '-sk', t], capture_output=True, text=True).stdout.split()[0]); shutil.rmtree(t)
    with open(ROOT + '/deletions.log', 'a') as f: f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  deleted {t}  {kb / 1048576:.2f} GB  (verified rows + summary written first)\n')
    return kb / 1048576


def keep_one_trunk(p):
    """Copy ONE trunk per complex to kept_trunks/ before the rest are deleted (user request 2026-09-29, for checking the setup later):
    the seed of donor A00 (the top-ipTM pose), so it can be reloaded alongside a result we already have. Verified by loading it back."""
    import numpy as np
    seed = None
    for l in open(f'{R(p)}/select/queue_ownseed.tsv'):
        f = l.rstrip('\n').split('\t')
        if f[3] == 'A00': seed = f[4]; break
    assert seed, ('no A00 donor', p)
    src = f'{R(p)}/trunks/{p}/free_{seed}.npz'; dst_dir = f'{ROOT}/kept_trunks/{p}'; dst = f'{dst_dir}/free_{seed}.npz'
    assert os.path.exists(src), src
    os.makedirs(dst_dir, exist_ok=True); shutil.copy2(src, dst)
    a, b = np.load(src), np.load(dst)
    assert str(b['dtype']) == 'bf16' and b['z_trunk'].shape == a['z_trunk'].shape and (a['z_trunk'] == b['z_trunk']).all() and (a['s_trunk'] == b['s_trunk']).all(), ('kept trunk copy does not match', dst)
    with open(ROOT + '/kept_trunks/KEPT.log', 'a') as f: f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {p}  seed {seed}  {os.path.getsize(dst) / 1e6:.0f} MB  {dst}\n')
    return dst


def finalize(p, roles):
    probs, met = health(p, roles)
    assert not probs, ('HEALTH CHECK FAILED, trunks kept', probs)
    rows = summarize(p, roles, met)
    assert rows[0]['dropout_true'] == 0, ('stage C ran with MC dropout ON for some folds; expected none', p, rows[0]['dropout_true'])
    kept = keep_one_trunk(p); gb = delete_trunks(p)
    json.dump(dict(finished=time.strftime('%Y-%m-%d %H:%M:%S'), trunks_deleted_gb=round(gb, 2), kept_trunk=kept, verified={k: v['verified'] for k, v in met.items()}, summary=rows), open(f'{R(p)}/DONE.json', 'w'), indent=1)
    log(f'DONE {p}: {len(rows)} interfaces summarized, trunks deleted ({gb:.1f} GB): ' + '; '.join(f"{x['interface']} free {x['free_pick_dockq']:.2f}{'*' if x['free_acceptable'] else ''} | ctl {x['ctl_pick_dockq']:.2f}{'*' if x['ctl_acceptable'] else ''} | templated {x['arm_pick_dockq']:.2f}{'*' if x['arm_acceptable'] else ''}" for x in rows))


# ---------------------------------------------------------------- controller
def quota_used_gb():
    try:
        line = subprocess.run(['quota', '-s'], capture_output=True, text=True).stdout.strip().split('\n')[-1].split(); v = line[1]
        return float(v[:-1]) * (1024 if v[-1] in 'Tt' else 1)
    except Exception: return 0.0


def prior_groups(roles, tok):
    """Class of each complex from the ORIGINAL FoldBench free pool (seeds 101-150, pool.csv): an interface is 'decoy' when its top-ipTM pose is wrong (DockQ < 0.23)
    but an acceptable pose exists among the 250, 'solved' when the top-ipTM pose is acceptable, else 'nocov'. A complex with any decoy interface is 'decoy'."""
    pool = collections.defaultdict(list)
    for r in csv.DictReader(open(FB + '/pool.csv')):
        if 101 <= int(r['seed']) <= 150: pool[(r['pdb'], r['chain1'], r['chain2'])].append((float(r['pair_iptm']), float(r['dockq'])))
    grp = {}
    for p in tok:
        cs = []
        for x in roles[p]:
            R = pool[(p, x['chain_1'], x['chain_2'])]
            if R: cs.append('solved' if max(R)[1] >= 0.23 else ('decoy' if max(d for _, d in R) >= 0.23 else 'nocov'))
        grp[p] = 'decoy' if 'decoy' in cs else ('solved' if 'solved' in cs else ('nocov' if 'nocov' in cs else 'unknown'))
    return grp


def pick_job(order, st, tok, a):
    """Priority: finish stage C work (frees trunks) > keep stage A going > start the next complex (smallest first) if it fits the cap and quota."""
    for p in order:
        if st[p] == 'C':
            for kind in ('T', 'K'):
                if joinable(p, 'C', kind) > 0: return ('C', p, kind)
    for p in order:
        if st[p] == 'A' and joinable(p, 'A') > 0: return ('A', p, None)
    inflight = sum(trunk_gb(tok[p]) for p in order if st[p] in ('A', 'B', 'C', 'D'))
    for p in order:
        if st[p] == 'todo':
            if inflight + trunk_gb(tok[p]) <= a.cap_gb and quota_used_gb() + trunk_gb(tok[p]) + 60 < QUOTA_LIMIT_GB:
                os.makedirs(f'{R(p)}/free', exist_ok=True); open(f'{R(p)}/queue_free.tsv', 'w').write(f'free\t{p}\t*\tfree\t{SEEDS[0]}-{SEEDS[-1]}\n')
                st[p] = 'A'; log(f'starting complex {p} ({tok[p]} tokens, est trunks {trunk_gb(tok[p]):.1f} GB, in flight {inflight:.0f} GB)'); return ('A', p, None)
            break   # smallest-first: never skip ahead of a complex that does not fit yet
    return None


def stop_holder(g, s):
    s['proc'].terminate()
    try: s['proc'].wait(timeout=15)
    except Exception: s['proc'].kill()
    log(f'gpu{g} holder released')


def controller(a, roles, tok):
    gpus = [int(g) for g in a.gpus.split(',')]; order = sorted((t, p) for p, t in tok.items() if p not in DONE_ALREADY and (not a.only or p in a.only.split(',')))
    grp = prior_groups(roles, tok); rank = {'decoy': 0, 'solved': 1, 'nocov': 2, 'unknown': 3}
    order = [p for p in sorted((p for _, p in order), key=lambda p: (rank[grp[p]], tok[p]))]   # decoy complexes first, then solved, then no-acceptable; smallest first within each
    slots = {g: None for g in gpus}; bproc = {}; attempts = collections.Counter(); last = 0
    busy = lambda: any(x and not x.get('hold') for x in slots.values())
    log(f'controller start: {len(order)} complexes, gpus {gpus}, cap {a.cap_gb} GB, groups {dict(collections.Counter(grp[p] for p in order))}, order {[(p, grp[p]) for p in order[:8]]}...')
    while True:
        st = {p: status(p, roles) for p in order}
        for g, s in slots.items():                                    # reap GPU workers / holders
            if s and s['proc'].poll() is not None:
                if s.get('hold'): slots[g] = None; continue
                rc = s['proc'].returncode; s['log'].close(); log(f'gpu{g} {s["stage"]}:{s["pdb"]} exited rc={rc}'); slots[g] = None
                p, stage = s['pdb'], s['stage']
                still = any(x and not x.get('hold') and x['pdb'] == p and x['stage'] == stage for x in slots.values())
                if not still and status(p, roles) == stage:
                    attempts[(p, stage)] += 1
                    if attempts[(p, stage)] >= 4: park(p, f'stage {stage} incomplete after {attempts[(p, stage)]} attempts')
        for p, (pr, lf) in list(bproc.items()):                        # reap CPU stage B
            if pr.poll() is not None:
                lf.close(); del bproc[p]
                if pr.returncode != 0: park(p, f'stage B failed (rc={pr.returncode}); see {R(p)}/stageB.log')
                else: log(f'stage B ok for {p}: {open(R(p) + "/select/B_OK").read().strip()}')
        st = {p: status(p, roles) for p in order}
        for p in order:
            if st[p] == 'D':
                try: finalize(p, roles)
                except Exception as e: park(p, f'finalize failed: {e!r} (trunks NOT deleted)')
        st = {p: status(p, roles) for p in order}
        stop = os.path.exists(ROOT + '/STOP')
        if not stop:
            for p in order:
                if st[p] == 'B' and p not in bproc and len(bproc) < a.max_b and not any(x and not x.get('hold') and x['pdb'] == p and x['stage'] == 'A' for x in slots.values()):
                    lf = open(f'{R(p)}/stageB.log', 'a'); log(f'starting stage B (CPU) for {p}')
                    bproc[p] = (subprocess.Popen([PY, os.path.abspath(__file__), 'stageB', p], stdout=lf, stderr=subprocess.STDOUT), lf)
            for g in gpus:
                s = slots[g]
                if s is not None and not s.get('hold'): continue
                job = pick_job(order, st, tok, a)
                if job:
                    if s: stop_holder(g, s); slots[g] = None
                    stage, p, kind = job; cmd, env = gpu_cmd(stage, p, g, kind or 'T'); os.makedirs(R(p), exist_ok=True)
                    lf = open(f'{R(p)}/gpu_{stage}{kind or ""}.log', 'a'); log(f'gpu{g} -> {stage}{kind or ""}:{p}')
                    slots[g] = dict(proc=subprocess.Popen(cmd, cwd=H, env=env, stdout=lf, stderr=subprocess.STDOUT), pdb=p, stage=stage, log=lf)
                elif s is None:                                        # nothing to run: hold the card so it is not lost, and keep looking
                    pr = subprocess.Popen([PY, f'{H}/hold_gpus.py', '--gpus', str(g), '--gb', '1'], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    slots[g] = dict(proc=pr, hold=True, pdb=None, stage='hold', log=None); log(f'gpu{g} has no work: holder started')
        if time.time() - last > 300:
            last = time.time(); c = collections.Counter(st.values())
            log('status ' + ' '.join(f'{k}={v}' for k, v in sorted(c.items())) + ' | slots ' + ' '.join(f'g{g}:{("hold" if s.get("hold") else s["stage"] + ":" + s["pdb"]) if s else "idle"}' for g, s in slots.items()) + f' | B running {list(bproc)}')
        done_all = all(v in ('done', 'parked') for v in st.values())
        if (done_all or stop) and not bproc and not busy():
            for g, s in slots.items():
                if s and s.get('hold'): stop_holder(g, s)
            log('all complexes done or parked; exiting' if done_all else 'STOP file: exiting'); break
        time.sleep(10)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--gpus', default='0,1,2,5'); r.add_argument('--cap-gb', type=float, default=450); r.add_argument('--only', default=''); r.add_argument('--max-b', type=int, default=2)
    b = sub.add_parser('stageB'); b.add_argument('pdb'); sub.add_parser('status')
    a = ap.parse_args(); roles, tok = load_info()
    if a.cmd == 'stageB': stageB(a.pdb, roles)
    elif a.cmd == 'status':
        c = collections.Counter()
        for p in sorted(tok, key=tok.get):
            if p in DONE_ALREADY: continue
            s = status(p, roles); c[s] += 1
            if s != 'todo': print(f'{p:6s} {tok[p]:5d} tokens  {s}')
        print(dict(c))
    else: controller(a, roles, tok)


if __name__ == '__main__': main()
