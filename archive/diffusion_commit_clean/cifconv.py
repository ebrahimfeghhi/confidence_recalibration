# Convert a Protenix-output cif into a template-parseable cif.
import sys
from io import StringIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
def convert(src_text, tid):
    d=MMCIF2Dict(StringIO(src_text))
    # unique (label_asym_id, label_entity_id) from atom_site, preserving order
    aid=d.get('_atom_site.label_asym_id',[]); eid=d.get('_atom_site.label_entity_id',[])
    seen=[]; pairs=[]
    for a,e in zip(aid,eid):
        if a not in seen: seen.append(a); pairs.append((a,e))
    text=src_text.replace('PEPTIDE LINKING','peptide linking')
    lines=text.splitlines(); out=[]
    for ln in lines:
        out.append(ln)
        if ln.startswith('data_'):
            out.append('_pdbx_audit_revision_history.revision_date   2000-01-01')
    # append struct_asym loop
    out.append('loop_'); out.append('_struct_asym.id'); out.append('_struct_asym.entity_id')
    for a,e in pairs: out.append(f'{a} {e}')
    out.append('#')
    return '\n'.join(out)+'\n'
if __name__=='__main__':
    src=sys.argv[1]; tid=sys.argv[2]; dst=sys.argv[3]
    open(dst,'w').write(convert(open(src).read(),tid))
    print('wrote',dst)
