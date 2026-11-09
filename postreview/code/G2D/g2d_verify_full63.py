#!/usr/bin/env python3
"""Independent fold-level comparison: old D (A), new D (B), restored R (C), for 4 estimators on all 63 tasks."""
import argparse,csv,gzip,hashlib,io,json,math,statistics,zipfile
from collections import defaultdict
from pathlib import Path
from itertools import combinations
ALGOS=('LR','SVC','RF','kNN')
MEMBER='VersionShift/results/STAGE_C_CPU_COMPLETION/merged/POS.csv.gz'
OLD='0.21.3'; NEW='0.22.2.post1'
def fold_key(row):
    def int_id(x):
        val=float(x)
        if not val.is_integer(): raise ValueError('non-integral fold index')
        return int(val)
    return (row['dataset'],row['algo'],int_id(row['rep']),int_id(row['fold']))
def load_baseline(zpath):
    rows={'A':{},'B':{}}
    with zipfile.ZipFile(zpath) as z:
        with gzip.GzipFile(fileobj=io.BytesIO(z.read(MEMBER))) as g:
            for row in csv.DictReader(io.TextIOWrapper(g)):
                if row['policy']!='D' or row['status']!='ok' or row['algo'] not in ALGOS: continue
                side='A' if row['release']==OLD else 'B' if row['release']==NEW else None
                if side is None:continue
                key=fold_key(row)
                if key in rows[side]: raise ValueError('duplicate baseline '+repr((side,key)))
                rows[side][key]=(float(row['ba']),row['pred_sha'])
    return rows
def load_restored(paths):
    rows={}
    for path in paths:
        with open(path,newline='') as f:
            for row in csv.DictReader(f):
                key=fold_key(row)
                if key in rows: raise ValueError('duplicate restored '+repr(key))
                rows[key]=(float(row['ba']),row['pred_sha'])
    return rows
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zip',required=True);p.add_argument('--pilot',required=True);p.add_argument('--holdout',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    base=load_baseline(a.zip);C=load_restored([a.pilot,a.holdout])
    D={**base,'C':C}
    if not (set(D['A'])==set(D['B'])==set(D['C'])):raise SystemExit('FAIL: missing or extra fold identities '+repr({q:len(v) for q,v in D.items()}))
    if len(C)!=63*4*30:raise SystemExit('FAIL: expected 7560 unique restored folds')
    score_bad=[];hash_bad=[]
    for key in sorted(C):
        va,ha=D['A'][key];vc,hc=D['C'][key]
        if abs(va-vc)>1e-12:score_bad.append([key,va,vc])
        if ha!=hc:hash_bad.append([key,ha,hc])
    means={}
    grouped=defaultdict(list)
    for side,rows in D.items():
        for (ds,algo,r,f),(score,_) in rows.items():grouped[side,ds,algo].append(score)
    for key,vals in grouped.items():
        if len(vals)!=30:raise ValueError('expected exactly 30 folds: '+repr(key))
        means[key]=statistics.mean(vals)*100
    per_dataset=[]
    for ds in sorted(set(k[0] for k in C)):
        entry={'dataset':ds}
        for left,right,label in [('A','B','AB'),('A','C','AC'),('C','B','CB')]:
            shifts=[]
            for aa,bb in combinations(ALGOS,2):
                diff=(means[right,ds,aa]-means[right,ds,bb])-(means[left,ds,aa]-means[left,ds,bb])
                shifts.append(abs(diff))
            entry['M_'+label+'_pp']=statistics.mean(shifts)
            wa=max(ALGOS,key=lambda alg:means[left,ds,alg]);wb=max(ALGOS,key=lambda alg:means[right,ds,alg])
            entry['winner_changed_'+label]=(wa!=wb)
        per_dataset.append(entry)
    result={'status':'PASS' if not score_bad and not hash_bad else 'FAIL',
            'source_archive_sha256':hashlib.sha256(Path(a.zip).read_bytes()).hexdigest(),
            'matched_folds':len(C),'score_mismatches':len(score_bad),'prediction_hash_mismatches':len(hash_bad),
            'first_score_mismatches':score_bad[:10],'first_hash_mismatches':hash_bad[:10],
            'scope':'63 datasets, four estimators, 30 frozen folds each',
            'means_pp':{q:statistics.mean([d['M_'+q+'_pp'] for d in per_dataset]) for q in ('AB','AC','CB')},
            'winner_changes':{q:sum(d['winner_changed_'+q] for d in per_dataset) for q in ('AB','AC','CB')}}
    (out/'G2D_full63_verification.json').write_text(json.dumps(result,indent=2))
    with (out/'G2D_dataset_comparisons.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=per_dataset[0].keys());w.writeheader();w.writerows(per_dataset)
    print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise SystemExit('FAIL: exact restoration is not supported for all folds')
if __name__=='__main__':main()
