#!/usr/bin/env python3
"""Independent descriptive six-estimator E2 comparison: original D and five E, new HGB E2.
Does not recalculate confidence intervals or confer confirmatory status.
"""
import argparse,csv,gzip,hashlib,io,itertools,json,statistics,zipfile
from collections import defaultdict
from pathlib import Path
ALGOS=('LR','SVC','kNN','RF','GB','HGB')
PAIRS=list(itertools.combinations(ALGOS,2))
RELEASES=('0.21.3','0.22.2.post1')
MEMBER='VersionShift/results/STAGE_C_CPU_COMPLETION/merged/POS.csv.gz'
def load(a):
    by_cell=defaultdict(dict)
    with zipfile.ZipFile(a.zip) as z:
        with gzip.GzipFile(fileobj=io.BytesIO(z.read(MEMBER))) as gz:
            for r in csv.DictReader(io.TextIOWrapper(gz)):
                if r['status']!='ok' or r['policy'] not in ('D','E') or r['release'] not in RELEASES:continue
                # Skip the old failed HGB/E arm; handled only by corrected E2.
                if r['algo']=='HGB' and r['policy']=='E':continue
                ds,algo,ver,pol=r['dataset'],r['algo'],r['release'],r['policy']
                fold=(int(float(r['rep'])),int(float(r['fold'])))
                k=(ds,algo,ver,pol)
                if fold in by_cell[k]:raise ValueError('Duplicate baseline fold: %r %r'%(k,fold))
                by_cell[k][fold]=float(r['ba'])
    expected=set((r,f) for r in range(6) for f in range(5))
    with open(a.hgb, newline='') as f:
        for r in csv.DictReader(f):
            if r['policy']!='E2' or r['algo']!='HGB' or r['release'] not in RELEASES:raise ValueError('Unexpected HGB E2 row')
            k=(r['dataset'],'HGB',r['release'],'E2')
            fold=(int(float(r['rep'])),int(float(r['fold'])))
            if fold in by_cell[k]:raise ValueError('Duplicate E2 fold: %r %r'%(k,fold))
            by_cell[k][fold]=float(r['ba'])
    for k,folds in by_cell.items():
        if k[3]=='E2' and set(folds)!=expected:raise ValueError('Incomplete HGB E2: %r'% (k,))
    datasets=sorted({k[0] for k in by_cell if k[3]=='E2'})
    for ds in datasets:
        for algo in ALGOS:
            for ver in RELEASES:
                for p in ('D', 'E2' if algo=='HGB' else 'E'):
                    k=(ds,algo,ver,p)
                    if set(by_cell.get(k,{}))!=expected:raise ValueError('Missing/mismatched cell: %r size=%d'%(k,len(by_cell.get(k,{}))))
    means={k:sum(folds.values())/30*100 for k,folds in by_cell.items() if k[0] in datasets}
    def treatment(ds,algo,ver,mode):return means[ds,algo,ver,'D' if mode=='D' else ('E2' if algo=='HGB' else 'E')]
    def calc(mode,scope):
        mags=[];change=0;over=0;winner=[]
        for ds in scope:
            old={algo:treatment(ds,algo,RELEASES[0],mode) for algo in ALGOS}
            new={algo:treatment(ds,algo,RELEASES[1],mode) for algo in ALGOS}
            m=sum(abs((new[x]-new[y])-(old[x]-old[y])) for x,y in PAIRS)/len(PAIRS)
            mags.append(m);over+=m>0.5
            ow=max(ALGOS,key=lambda name:old[name]);nw=max(ALGOS,key=lambda name:new[name])
            winner.append((ds,ow,nw,m));change+=ow!=nw
        return dict(K=len(scope),M_pp=sum(mags)/len(mags),affected_gt_0_5=over,winner_changes=change,
                    dataset_results=[dict(dataset=ds,old_winner=ow,new_winner=nw,M_pp=m) for ds,ow,nw,m in winner])
    pilot=json.loads(Path(a.selection).read_text())['stage_b_datasets'];pilot=set(pilot)
    frames={'pilot16':sorted(set(datasets)&pilot),'holdout47':sorted(set(datasets)-pilot),'full63':datasets}
    comparisons={frame:{mode:calc(mode,items) for mode in ('D','E2')} for frame,items in frames.items() if items}
    # The full63 default counts should match independently certified baseline.
    if 'full63' in comparisons:
        d=comparisons['full63']['D']
        if abs(d['M_pp']-5.006712451138973)>1e-8 or d['winner_changes']!=25 or d['affected_gt_0_5']!=59:
            raise ValueError('Original D baseline mismatch; STOP rather than publish')
    report=dict(status='PASS',scope=list(frames.keys()),description='Post-review E2; all six estimators, identical dataset and fold identities',
                treatment='HGB corrected n_iter_no_change=None; remaining five estimators original policy E',
                original_zip_sha256=hashlib.sha256(Path(a.zip).read_bytes()).hexdigest(),
                comparisons=comparisons,limitations=['Post-review corrected map, not frozen v2.1 E policy','No CIs or tests recomputed','A complete fitted-parameter/provenance audit remains required'])
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'G3E_six_estimator_results.json').write_text(json.dumps(report,indent=2))
    with (out/'G3E_summary.txt').open('w') as f:
        for frame,contrasts in comparisons.items():
            print(frame,file=f)
            for mode,m in contrasts.items():print('  %s M=%.8f pp affected=%d/%d changed_winners=%d'%(mode,m['M_pp'],m['affected_gt_0_5'],m['K'],m['winner_changes']),file=f)
    print((out/'G3E_summary.txt').read_text())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--hgb',required=True)
    p.add_argument('--selection',required=True);p.add_argument('--out',required=True)
    load(p.parse_args())
