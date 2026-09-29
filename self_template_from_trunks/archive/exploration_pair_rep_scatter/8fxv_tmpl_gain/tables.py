import csv,collections,statistics as st
D='/home/ebrahim/tts_abag/pair_rep_scatter/8fxv_tmpl_gain'; NS='/home/ebrahim/tts_abag/experimental_data/protenix/foldbench'
def L(p,ifc='8fxv_AB'):
    d=collections.defaultdict(list)
    for r in csv.DictReader(open(p)):
        if r['iface']==ifc: d[r['arm']].append((float(r['iptm']),float(r['dockq']),int(r['sample'])))
    return d
O=L(NS+'/self_template_fixed_50_seed/results.csv');G1=L(D+'/g1/results.csv');G2=L(D+'/g2/results.csv')
print('folds orig/g1/g2:',len(O),len(G1),len(G2),'| 5 samples each g1/g2:',all(len(v)==5 for v in G1.values()),all(len(v)==5 for v in G2.values()),'| same donors:',sorted(G1)==sorted(G2))
S1={r['tag']:r for r in csv.DictReader(open(D+'/fold_scatter_g1.csv'))};S2={r['tag']:r for r in csv.DictReader(open(D+'/fold_scatter_g2.csv'))}
print('| donor | donor DockQ to crystal | free-fold agreement | agreement g=1 | agreement g=2 | ipTM mean of 5 g=1 (min-max) | ipTM mean of 5 g=2 (min-max) | pose DockQ median: original / g=1 / g=2 |');print('|---|---|---|---|---|---|---|---|')
for t in sorted(G2):
    a=[x[0] for x in G1[t]];b=[x[0] for x in G2[t]];m=lambda d:st.median(x[1] for x in d[t])
    print(f"| {t} | {float(S1[t]['donor_dockq']):.2f} | {float(S1[t]['free_agreement']):.2f} | {float(S1[t]['tmpl_agreement']):.2f} | {float(S2[t]['tmpl_agreement']):.2f} | {st.mean(a):.3f} ({min(a):.3f}-{max(a):.3f}) | {st.mean(b):.3f} ({min(b):.3f}-{max(b):.3f}) | {m(O):.2f} / {m(G1):.2f} / {m(G2):.2f} |")
for n,d in (('orig',O),('g1',G1),('g2',G2)):
    b=max((v[0],v[1],t) for t,dd in d.items() for v in dd);allq=[v[1] for dd in d.values() for v in dd]
    top=sorted(((round(v[0],3),round(v[1],2),t) for t,dd in d.items() for v in dd),reverse=True)[:4]
    print(n,'top-ipTM pose: ipTM %.3f DockQ %.3f donor %s'%b,'| share of poses >=0.23: %.2f'%(sum(q>=.23 for q in allq)/len(allq)),'| top4 (ipTM,DockQ,donor):',top)
print('median agreement: free %.3f g1 %.3f g2 %.3f'%(st.median(float(r['free_agreement']) for r in S1.values()),st.median(float(r['tmpl_agreement']) for r in S1.values()),st.median(float(r['tmpl_agreement']) for r in S2.values())))
