"""Prospective task allocation using original task IDs and metadata, no results."""
import argparse, json, hashlib, csv
from pathlib import Path
import numpy as np
from .utils import read_json,sha256,write_once,require

def stratified_split(tasks, n_pilot=15, seed=20261116):
    require(len(tasks)==63,'exactly 63 eligible prior tasks required')
    ids=[int(x['task_id']) for x in tasks]
    require(len(set(ids))==63,'duplicate task IDs')
    rng=np.random.RandomState(seed)
    strata={}
    for x in tasks:
        n=int(x['n']);k=int(x['k']);p=int(x['p'])
        # Fixed, outcome-independent strata. Original n/p are descriptive metadata.
        key=('binary' if k==2 else 'multi','small' if n<=2000 else 'large')
        strata.setdefault(key,[]).append(x)
    for key,v in strata.items():v.sort(key=lambda x:int(x['task_id']));rng.shuffle(v)
    # largest-remainder allocation with minimum feasible floor, deterministically resolved
    sizes={k:len(v) for k,v in strata.items()}
    quota={k:15*n/63 for k,n in sizes.items()}
    alloc={k:int(np.floor(q)) for k,q in quota.items()}
    remaining=15-sum(alloc.values())
    order=sorted(strata,key=lambda k:(-(quota[k]-alloc[k]),str(k)))
    for k in order[:remaining]:alloc[k]+=1
    pilot=[];hold=[]
    for k in sorted(strata):
        pilot.extend(int(v['task_id']) for v in strata[k][:alloc[k]])
        hold.extend(int(v['task_id']) for v in strata[k][alloc[k]:])
    require(len(pilot)==15 and len(hold)==48 and len(set(pilot+hold))==63,'allocation invalid')
    return sorted(pilot),sorted(hold),{str(k):{'population':sizes[k],'pilot':alloc[k]} for k in sorted(strata)}

def recover_from_legacy(legacy):
    x=read_json(legacy)
    ret=[]
    for d in x['datasets']:
        i=d.get('info',{})
        if i.get('origin')!='OpenML-CC18':raise RuntimeError('unknown source for task '+str(d))
        tid=i.get('task')
        require(isinstance(tid,int),'original task id not recorded for '+d['name'])
        require(d['name']=='openml_%d'%tid,'original task identity mismatch')
        require(isinstance(i.get('dataset_id'),int) and isinstance(i.get('version'),int), 'original dataset revision missing')
        ret.append({'task_id':tid,'dataset_id':i['dataset_id'],'dataset_version':i['version'],
                    'n':int(d['n']),'p':int(d['p']),'k':int(d['k']),
                    'legacy_dataset_sha256':d['dataset_sha256'],'legacy_folds_sha256':d['folds_sha256'],
                    'legacy_dataset_name':d['name'],'legacy_seeds':d['seeds'],'dataset_name':i.get('dataset_name','')})
    require(len(ret)==63 and len(set(v['task_id'] for v in ret))==63,'legacy frame not 63 unique tasks')
    return ret

def read_catalog(path):
    with Path(path).open(newline='') as f: rows=list(csv.DictReader(f))
    cols={'task_id','dataset_id','dataset_version','n','p_encoded','k','min_class_count','eligible'}
    require(rows and cols.issubset(rows[0]),'external metadata catalog missing required columns')
    return rows

def external_sample(rows,core,seed,limit=20):
    ids=set(x['task_id'] for x in core);dids=set(x['dataset_id'] for x in core)
    aliases={str(x.get('dataset_name','')).strip().casefold() for x in core if x.get('dataset_name')}
    good=[];exclusions=[]; seenids=set();seen_dids=set()
    for x in sorted(rows,key=lambda x:int(x['task_id'])):
        try:
            tid=int(x['task_id']);did=int(x['dataset_id']); n=int(x['n']);p=int(x['p_encoded']);
            k=int(x['k']);m=int(x['min_class_count']);el=x['eligible'].strip().lower()
            reason=None
            if tid in ids or did in dids or (x.get('dataset_name','').strip().casefold() in aliases and x.get('dataset_name','').strip()):
                reason='CC18_overlapping_task_or_dataset' 
            elif tid in seenids or did in seen_dids:reason='duplicated_external_task_or_dataset'
            elif el not in ('1','true','yes'):reason='not_eligible_in_frozen_catalog'
            elif min(n,m,p,k)<=0 or k<2 or m<10 or p>500:reason='violates_predeclared_rules'
            if reason:exclusions.append({'task_id':tid,'reason':reason});continue
            good.append({'task_id':tid,'dataset_id':did,'dataset_version':int(x['dataset_version']),
                         'n':min(n,5000),'p':p,'k':k})
            seenids.add(tid);seen_dids.add(did)
        except (ValueError,KeyError) as e:raise RuntimeError('bad external metadata row '+str(x)) from e
    # deterministic stratified ordering, take 20, no outcome-based refill
    if len(good)<limit:return [],exclusions,{'status':'INSUFFICIENT_CANDIDATES','eligible':len(good)}
    rng=np.random.RandomState(seed)
    strata={}
    for x in good:
        key=(x['k']==2,x['n']>2000)
        strata.setdefault(key,[]).append(x)
    for s in strata.values():rng.shuffle(s)
    selected=[]
    while len(selected)<limit and any(strata.values()):
        for k in sorted(strata):
            if strata[k] and len(selected)<limit:selected.append(strata[k].pop())
    return selected,exclusions,{'status':'READY','eligible':len(good)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy-plan',required=True);p.add_argument('--external-catalog',required=True)
    p.add_argument('--seed',type=int,default=20261116);p.add_argument('--out',required=True);a=p.parse_args()
    core=recover_from_legacy(a.legacy_plan)
    pilot,held,strata=stratified_split(core,seed=a.seed)
    external,exclusions,status=external_sample(read_catalog(a.external_catalog),core,a.seed)
    freeze={'status':'PROPOSED_NOT_FROZEN','legacy_plan_sha256':sha256(a.legacy_plan),
            'catalog_sha256':sha256(a.external_catalog),'seed':a.seed,'pilot_task_ids':pilot,
            'evaluation_task_ids':held,'external_task_ids':[r['task_id'] for r in external],
            'original_tasks':core,'external_tasks':external,'external_exclusions':exclusions,
            'external_selection_status':status,'allocation_strata':strata,
            'warning':'metadata freeze does not authorize outcome analysis'}
    write_once(a.out,freeze)
    print('PLAN_DRAFT',len(pilot),len(held),len(external),'external',status['status'])
if __name__=='__main__':main()
