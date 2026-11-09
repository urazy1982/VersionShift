#!/usr/bin/env python3
"""G2-G read-only cross-release boosting comparisons.
GB: original old D vs new D. HGB: old D (256), old intervention (255), new D (255).
No causal mediation identification asserted; identical max_bins != identical runtime/config.
"""
import argparse, csv, gzip, hashlib, io, json, math, statistics, zipfile
from collections import defaultdict
from pathlib import Path
BASE='VersionShift/results/STAGE_C_CPU_COMPLETION/merged/POS.csv.gz'
ALGS=('GB','HGB')
RELS=('0.21.3','0.22.2.post1')
ARMS={'GB_NEW_PRESORT_AUTO':('GB','0.22.2.post1'), 'HGB_OLD_MAXBINS255':('HGB','0.21.3')}

def key(r):
    return (r['dataset'],r['algo'],int(float(r['rep'])),int(float(r['fold'])))
def score(r):
    v=float(r['ba'])
    if not math.isfinite(v) or not 0<=v<=1:raise ValueError('Invalid BA')
    if not r.get('pred_sha') or len(r['pred_sha'])!=16:raise ValueError('Invalid prediction hash')
    return v

def intervention(folder,filename):
    p=Path(folder)/filename
    with p.open(newline='') as f: entries=list(csv.DictReader(f))
    if not entries:raise ValueError('No rows in '+str(p))
    out={}
    for r in entries:
        arm=r['arm']; k=(arm,)+key(r)
        if arm not in ARMS or (r['algo'],r['release'])!=ARMS[arm]:raise ValueError('Wrong arm '+str(k))
        score(r)
        if k in out:raise ValueError('Duplicate '+str(k))
        out[k]=r
    return out

def summary(rows):
    ds=defaultdict(list)
    for (dataset,rep,fold), (a,b) in rows.items():
        ds[dataset].append((100*(b[0]-a[0]),a[1]!=b[1]))
    per=[]
    for d in sorted(ds):
        vals=ds[d]
        if len(vals)!=30:raise ValueError('Incomplete dataset '+d)
        per.append({'dataset':d,'mean_signed_pp':statistics.mean(x for x,_ in vals),
                    'mean_absolute_fold_pp':statistics.mean(abs(x) for x,_ in vals),
                    'hash_changed_folds':sum(h for _,h in vals),
                    'score_changed_folds':sum(abs(x)>1e-10 for x,_ in vals)})
    return {'datasets':len(ds),'folds':len(rows),
            'mean_signed_pp':statistics.mean(x['mean_signed_pp'] for x in per),
            'mean_absolute_fold_pp':statistics.mean(x['mean_absolute_fold_pp'] for x in per),
            'mean_absolute_dataset_mean_pp':statistics.mean(abs(x['mean_signed_pp']) for x in per),
            'hash_changed_folds':sum(x['hash_changed_folds'] for x in per),
            'score_changed_folds':sum(x['score_changed_folds'] for x in per),
            'datasets_with_mean_score_change':sum(abs(x['mean_signed_pp'])>1e-10 for x in per)},per

def main(a):
    z=Path(a.zip); out=Path(a.out)
    pilots=intervention(a.pilot,'G2E_FOLD_SCORES.csv')
    holdouts=intervention(a.holdout,'G2F_FOLD_SCORES.csv')
    if set(pilots)&set(holdouts):raise ValueError('Pilot/holdout overlap')
    inter={**pilots,**holdouts}
    if len(pilots)!=960 or len(holdouts)!=2820 or len(inter)!=3780:raise ValueError('Incorrect intervention coverage')
    ids={x[1] for x in inter}
    if len(ids)!=63:raise ValueError('Need 63 datasets')
    for name,folder in [('pilot',a.pilot),('holdout',a.holdout)]:
        namefile='G2E_SUMMARY.json' if name=='pilot' else 'G2F_SUMMARY.json'
        s=json.loads((Path(folder)/namefile).read_text())
        if s['status']!='PASS':raise ValueError(name+' not PASS')
        if name=='pilot':pilot_summary=s
        else:
            if s['worker_sha256']!=pilot_summary['worker_sha256'] or s['plan_sha256']!=pilot_summary['plan_sha256']:
                raise ValueError('Pilot/holdout source or plan differs')
    baseline={}
    with zipfile.ZipFile(z) as f:
        with gzip.GzipFile(fileobj=io.BytesIO(f.read(BASE))) as g:
            for r in csv.DictReader(io.TextIOWrapper(g)):
                if r['status']!='ok' or r['policy']!='D' or r['algo'] not in ALGS or r['dataset'] not in ids:continue
                if r['release'] not in RELS:continue
                k=(r['release'],)+key(r)
                if k in baseline:raise ValueError('Duplicate baseline '+str(k))
                baseline[k]=(score(r),r['pred_sha'])
    if len(baseline)!=63*2*2*30:raise ValueError('Baseline incomplete: '+str(len(baseline)))
    if len([k for k in inter if k[0]=='HGB_OLD_MAXBINS255'])!=1890:raise ValueError('HGB intervention incomplete')
    if len([k for k in inter if k[0]=='GB_NEW_PRESORT_AUTO'])!=1890:raise ValueError('GB intervention incomplete')
    comparisons=defaultdict(dict)
    for d in ids:
        for rep in range(6):
            for fold in range(5):
                for algo in ALGS:
                    k=(d,algo,rep,fold); unit=(d,rep,fold)
                    old=baseline[('0.21.3',)+k]; new=baseline[('0.22.2.post1',)+k]
                    comparisons[algo+'_oldD_to_newD'][unit]=(old,new)
                    if algo=='HGB':
                        r=inter[('HGB_OLD_MAXBINS255',)+k]
                        altered=(score(r),r['pred_sha'])
                        comparisons['HGB_old256_to_old255'][unit]=(old,altered)
                        comparisons['HGB_old255_to_new255'][unit]=(altered,new)
                    else:
                        r=inter[('GB_NEW_PRESORT_AUTO',)+k]
                        altered=(score(r),r['pred_sha'])
                        comparisons['GB_newD_to_presort_auto'][unit]=(new,altered)
    report={'status':'PASS','scope':'63 datasets and 30 paired folds; cross-release boosting contrasts',
            'source_zip_sha256':hashlib.sha256(z.read_bytes()).hexdigest(),
            'plan_sha256':pilot_summary['plan_sha256'], 'worker_sha256':pilot_summary['worker_sha256'],
            'contrast_definitions':{
                'GB_oldD_to_newD':'Old shipped GB versus new shipped GB: total release shift',
                'GB_newD_to_presort_auto':'Within-new-runtime GB presort diagnostic',
                'HGB_oldD_to_newD':'Old shipped HGB (max_bins=256) versus new shipped HGB (max_bins=255): total release shift',
                'HGB_old256_to_old255':'Within-old-runtime HGB change in max_bins only',
                'HGB_old255_to_new255':'Old runtime with max_bins=255 versus new shipped HGB with max_bins=255: remaining cross-runtime difference; other configuration/implementation changes uncontrolled'},
            'contrasts':{},'limitations':[
                'Equal max_bins values do not by themselves imply equivalent full parameter configurations or implementations',
                'Absolute contrasts need not add; signed fold-level differences do add under paired fold keys',
                'The within-old-runtime max_bins change does not identify default-driven impact on the newer runtime',
                'All 63 datasets are the finite eligible frame; inference beyond the frame requires additional assumptions']}
    out.mkdir(parents=True,exist_ok=True)
    per_rows=[]
    for name, data in sorted(comparisons.items()):
        s,per=summary(data);report['contrasts'][name]=s
        for x in per:per_rows.append({'contrast':name,**x})
    # Paired score-contrast identity for HGB at fold level: (new-old256)=(old255-old256)+(new-old255).
    maxerr=0.0
    for k in comparisons['HGB_oldD_to_newD']:
        a,b=comparisons['HGB_oldD_to_newD'][k]
        aa,c=comparisons['HGB_old256_to_old255'][k]
        cc,bb=comparisons['HGB_old255_to_new255'][k]
        if a!=aa or b!=bb or c!=cc:raise ValueError('Contrast source mismatch '+str(k))
        maxerr=max(maxerr,abs((b[0]-a[0])-((c[0]-a[0])+(b[0]-c[0]))))
    report['HGB_signed_contrast_identity_max_error_pp']=100*maxerr
    with (out/'G2G_dataset_comparisons.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['contrast','dataset','mean_signed_pp','mean_absolute_fold_pp','hash_changed_folds','score_changed_folds']);writer.writeheader();writer.writerows(per_rows)
    (out/'G2G_analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({'status':report['status'],'contrasts':report['contrasts'],'identity_max_error_pp':report['HGB_signed_contrast_identity_max_error_pp']},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--pilot',required=True);p.add_argument('--holdout',required=True);p.add_argument('--out',required=True);main(p.parse_args())
