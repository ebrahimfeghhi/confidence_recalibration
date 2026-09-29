import csv,collections,statistics as st
P='/home/ebrahim/tts_abag/pair_rep_scatter'; D=P+'/fullmsa_nodropout'
NS='/home/ebrahim/tts_abag/experimental_data/protenix/foldbench/self_template_fixed_50_seed/results.csv'
def L(p,ifc):
    d=collections.defaultdict(list)
    for r in csv.DictReader(open(p)):
        if r['iface']==ifc: d[r['arm']].append((float(r['iptm']),float(r['dockq'])))
    return d
def S(p): return {r['tag']:r for r in csv.DictReader(open(p))}
for pdb,g1dir in (('9fww','9fww_tmpl_gain'),('8fxv','8fxv_tmpl_gain')):
    ifc=pdb+'_AB'; O=L(NS,ifc); G=L(f'{P}/{g1dir}/g1/results.csv',ifc); F=L(f'{D}/{pdb}/results.csv',ifc)
    SG=S(f'{P}/{g1dir}/fold_scatter_g1.csv'); SF=S(f'{D}/fold_scatter_{pdb}.csv')
    print(f'\n== {ifc}: folds original/g=1/full-MSA-no-dropout = {len(O)}/{len(G)}/{len(F)}; 5 samples each: {all(len(v)==5 for v in F.values())}; same donors: {sorted(G)==sorted(F)}')
    print('| donor | donor DockQ to crystal | agreement g=1 (default MSA+dropout) | agreement full MSA no dropout | ipTM mean of 5 (min-max) g=1 | ipTM mean of 5 (min-max) full MSA no dropout | pose DockQ median: original / g=1 / full MSA no dropout |'); print('|---|---|---|---|---|---|---|')
    for t in sorted(F):
        a=[x[0] for x in G[t]]; b=[x[0] for x in F[t]]; m=lambda d:st.median(x[1] for x in d[t])
        print(f"| {t} | {float(SG[t]['donor_dockq']):.2f} | {float(SG[t]['tmpl_agreement']):.2f} | {float(SF[t]['tmpl_agreement']):.2f} | {st.mean(a):.3f} ({min(a):.3f}-{max(a):.3f}) | {st.mean(b):.3f} ({min(b):.3f}-{max(b):.3f}) | {m(O):.2f} / {m(G):.2f} / {m(F):.2f} |")
    print('median agreement: free %.3f | g=1 %.3f | full MSA no dropout %.3f'%(st.median(float(r['free_agreement']) for r in SG.values()),st.median(float(r['tmpl_agreement']) for r in SG.values()),st.median(float(r['tmpl_agreement']) for r in SF.values())))
    for n,d in (('original arm',O),('g=1 default',G),('full MSA no dropout',F)):
        R=[(v[0],v[1],t) for t,dd in d.items() for v in dd]; top=max(R); w=max(r for r in R if r[1]<0.23); gd=[r for r in R if r[1]>=0.23]
        g=max(gd) if gd else None
        print(f'  {n:20s} top-ipTM pick: ipTM {top[0]:.3f} DockQ {top[1]:.2f} donor {top[2]} | best wrong ipTM {w[0]:.3f} | best good ipTM {g[0] if g else float("nan"):.3f} | gap {w[0]-(g[0] if g else float("nan")):+.3f} | share of poses >=0.23: {sum(r[1]>=.23 for r in R)/len(R):.2f}')
