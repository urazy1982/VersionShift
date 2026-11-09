#!/usr/bin/env python3
"""Read-only paired comparison of old defaults A, new defaults B, restoration C."""
import argparse, csv, gzip, hashlib, io, json, math, statistics, zipfile
from collections import defaultdict
from pathlib import Path

ARCHIVE_SHA = '5eb716b64c21ded6f979b0b1610bb68aee46ad2f24889a0d3b133bd430c6ff7d'
SOURCE = 'VersionShift/results/STAGE_C_CPU_COMPLETION/merged/POS.csv.gz'
ALGOS = ('LR', 'SVC', 'RF', 'kNN')
OLD, NEW = '0.21.3', '0.22.2.post1'

def mean_dict(rows):
    d = defaultdict(list)
    for ds, algo, rep, fold, value in rows:
        key = (ds, algo, int(float(rep)), int(float(fold)))  # merged file stores rep/fold as floats
        d[key].append(float(value))
    return d

def load_c(path):
    with open(path, newline='') as f:
        r=csv.DictReader(f)
        needed={'dataset','algo','rep','fold','ba','pred_sha'}
        assert needed.issubset(r.fieldnames), 'Restoration schema mismatch'
        return mean_dict((x['dataset'], x['algo'], x['rep'], x['fold'], x['ba']) for x in r)

def load_ab(zpath, selected):
    by_release = {OLD: [], NEW: []}
    with zipfile.ZipFile(zpath) as z:
        with gzip.GzipFile(fileobj=z.open(SOURCE)) as f:
            for x in csv.DictReader(io.TextIOWrapper(f)):
                if x['status'] != 'ok' or x['policy'] != 'D' or x['algo'] not in ALGOS or x['dataset'] not in selected:
                    continue
                if x['release'] in by_release:
                    by_release[x['release']].append((x['dataset'],x['algo'],x['rep'],x['fold'],x['ba']))
    return {name: mean_dict(rows) for name, rows in by_release.items()}

def consolidate(d, label, expected_ds):
    cells=defaultdict(dict)
    for (ds,a,r,f), vv in d.items():
        assert len(vv)==1, '%s duplicate %r' % (label,(ds,a,r,f))
        assert math.isfinite(vv[0]) and 0 <= vv[0] <= 1, '%s bad score %r' % (label,(ds,a,r,f))
        assert 0 <= r < 6 and 0 <= f < 5, '%s bad fold %r' % (label,(ds,a,r,f))
        cells[(ds,a)][(r,f)] = vv[0]
    assert len(cells)==len(expected_ds)*len(ALGOS), '%s cell coverage %s' % (label,len(cells))
    for ds in expected_ds:
        for a in ALGOS:
            folds = cells[(ds,a)]
            assert len(folds)==30, '%s fold coverage %s %s: %d' % (label,ds,a,len(folds))
            assert set(folds)=={(r,f) for r in range(6) for f in range(5)}, '%s fold identity' % label
    return {(ds,a): statistics.mean(v.values())*100 for (ds,a),v in cells.items()}

def compute(A,B,C,ds_list):
    pairs = [(a,b) for i,a in enumerate(ALGOS) for b in ALGOS[i+1:]]
    out=[]
    for ds in ds_list:
        scores={tag:{a:dat[(ds,a)] for a in ALGOS} for tag,dat in [('A',A),('B',B),('C',C)]}
        gaps={tag:{a+'-'+b:scores[tag][a]-scores[tag][b] for a,b in pairs} for tag in scores}
        shift={name:{p:gaps[y][p]-gaps[x][p] for p in gaps['A']} for name,x,y in [('AB','A','B'),('AC','A','C'),('CB','C','B')]}
        assert all(abs(shift['AB'][p]-shift['AC'][p]-shift['CB'][p])<1e-8 for p in gaps['A'])
        winners={tag:max(ALGOS,key=lambda a:scores[tag][a]) for tag in scores}
        out.append({'dataset':ds,'scores_pp':scores,'winner':winners,'M_AB_pp':statistics.mean(abs(v) for v in shift['AB'].values()),'M_AC_pp':statistics.mean(abs(v) for v in shift['AC'].values()),'M_CB_pp':statistics.mean(abs(v) for v in shift['CB'].values()),'pair_shift_pp':shift})
    results={'scope':'pilot16, four classifiers only; not comparable to published six-classifier M',
             'datasets':len(out),'algorithms':list(ALGOS),'pairs':len(pairs)}
    for k in ('AB','AC','CB'):
        results['M_'+k+'_pp']=statistics.mean(x['M_'+k+'_pp'] for x in out)
    for k,x,y in [('AB','A','B'),('AC','A','C'),('CB','C','B')]:
        results['winner_changes_'+k]=sum(t['winner'][x]!=t['winner'][y] for t in out)
    results['per_estimator_mean_score_change_pp']={a:{
        k:statistics.mean(t['scores_pp'][y][a]-t['scores_pp'][x][a] for t in out)
        for k,x,y in [('AB','A','B'),('AC','A','C'),('CB','C','B')]}
        for a in ALGOS}
    return results,out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zip',required=True)
    p.add_argument('--restored',required=True)
    p.add_argument('--summary',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    zpath=Path(args.zip)
    digest=hashlib.sha256(zpath.read_bytes()).hexdigest()
    assert digest == ARCHIVE_SHA, 'Baseline archive SHA256 mismatch; refusing to compare'
    meta=json.loads(Path(args.summary).read_text())
    assert meta['status']=='PASS' and meta['datasets']==16 and meta['jobs']==64 and meta['folds']==1920, 'G2 summary mismatch'
    assert meta['plan_sha256']=='523c42a3dabee44e66d43d538b228939b01fd7fe1b68285f542f0980906492f4', 'Frozen plan hash mismatch'
    C_raw=load_c(args.restored)
    ds=sorted({key[0] for key in C_raw})
    assert len(ds)==16, 'Expected 16 pilot datasets'
    ab=load_ab(zpath,set(ds))
    A=consolidate(ab[OLD],'A',ds); B=consolidate(ab[NEW],'B',ds); C=consolidate(C_raw,'C',ds)
    results, details=compute(A,B,C,ds)
    results['baseline_zip_sha256']=digest
    results['restoration_plan_sha256']=meta['plan_sha256']
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/'G2_ABC_summary.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
    (out/'G2_ABC_dataset_details.json').write_text(json.dumps(details,indent=2,sort_keys=True)+'\n')
    text='G2 ABC pilot, 4 estimators / 16 datasets\n' + '\n'.join('%s: %s'%(k,v) for k,v in results.items())+'\n'
    (out/'G2_ABC_summary.txt').write_text(text)
    print(text)

if __name__=='__main__':main()
