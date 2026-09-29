"""Build one chain script per GPU for the 8 arms (MSA on / single row x 1,3,5,10 recycles) x 4 complexes, balanced by estimated cost:
donors x measured mean fold seconds (stage A run.json) x recycles/10 + 60 s model load; greedy longest-first onto 4 GPUs."""
import glob, json, collections, os
B = '/home/ebrahim/tts_abag/pair_rep_scatter/batch2'; PY = '/home/ebrahim/Protenix/.venv/bin/python'
PDBS = ['7yxu', '7zoz', '8be3', '8ux6']; GPUS = [0, 1, 2, 5]
donors = {p: sum(1 for l in open(f'{B}/select/queue_{p}.tsv') if l.strip()) for p in PDBS}
fs = {}
for p in PDBS:
    v = [json.load(open(f)).get('fold_seconds', 0) for f in glob.glob(f'{B}/free/{p}/*/run.json')]; fs[p] = sum(v) / len(v)
jobs = []
for c in (1, 3, 5, 10):
    for kind in ('msa', 'row1'):
        for p in PDBS: jobs.append((donors[p] * fs[p] * c / 10 + 60, f'{kind}_c{c}', c, kind, p))
jobs.sort(reverse=True); load = {g: 0.0 for g in GPUS}; chain = {g: [] for g in GPUS}
for cost, arm, c, kind, p in jobs:
    g = min(load, key=load.get); load[g] += cost; chain[g].append((arm, c, kind, p))
def cmd(arm, c, kind, p):
    e = 'PROTENIX_MSA_NROWS=1 PROTENIX_MSA_STRATEGY=topk PROTENIX_LOG_MSA=1' if kind == 'row1' else ''
    os.makedirs(f'{B}/resume/{arm}', exist_ok=True)
    return (f'cd /home/ebrahim/tts_abag/diffusion_commit_clean_v2 && CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '
            f'TRUNK_LOAD_DIR={B}/trunks {e} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 {PY} folder.py --queue {B}/select/queue_{p}.tsv '
            f'--out {B}/resume/{arm} --csv {B}/resume/{arm}/results.csv --claim-dir {B}/resume/{arm}/claims --dataset foldbench --templates {B}/select/templates '
            f'--ncycle {c} --nsample 5 --dtype bf16 2>&1 | tee -a {B}/resume/{arm}/w${{GPU}}_{p}.log')
for g in GPUS:
    open(f'{B}/stageC_gpu{g}.sh', 'w').write(f'export GPU={g}\n' + '\n'.join(cmd(*j) for j in chain[g]) + '\n')
print('donors', donors, '| mean fold s', {p: round(s, 1) for p, s in fs.items()})
print('estimated minutes per GPU', {g: round(l / 60) for g, l in load.items()}, '| jobs per GPU', {g: len(chain[g]) for g in GPUS})
