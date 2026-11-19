"""Resumable 8-worker scheduler with fully frozen inputs and no overlapping jobs."""
import argparse,concurrent.futures as futures,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from .utils import read_json,sha256,write_once,digest_json,require
from .binding import load_settings,source_binding,frozen_source_rows

ALGORITHMS=['LR','SVC','kNN','RF','GB','HGB','LinearSVC','AdaBoost']

def verify_frozen_sources(project, freeze_record):
    """Fail closed if a prospectively frozen input/analysis source was edited."""
    hashes=freeze_record.get('source_hashes',{})
    require(hashes, 'frozen code/data inventory absent')
    for rel,want in hashes.items():
        if rel == 'selected_plan':
            path=project/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json'
        elif rel == 'G0_ARCHIVE_AUDIT.json':
            path=project/'reports/G0_ARCHIVE_AUDIT.json'
        elif rel == 'G1_COMMON_STACK_CHECK.json':
            path=Path.home()/'versionshift_v3_envs/COMMON_STACK_CHECK.json'
        elif rel == 'author_approval_sha256':
            # The approval file location is author supplied at freeze; tracked by
            # the frozen hash and retained with the freeze evidence, not guessed here.
            continue
        elif '/' in rel:
            path=project/rel
        else:
            raise RuntimeError('unrecognized frozen evidence key '+rel)
        require(path.is_file(), 'frozen evidence missing '+str(path))
        require(sha256(path)==want,'frozen evidence changed '+str(path))
    if 'runs/plan/HISTORICAL_ENVIRONMENT_READINESS_v305.json' in hashes:
        from .historical_envs import validate_historical_readiness
        validate_historical_readiness(project)
    return True


def jobs_for_edge(project,plan,edge,frame,prep,policies,releases,result_dir,freeze_digest):
    edges=read_json(project/'config/edges.json')
    match=next((x for x in edges['new_primary_edges'] if x['id']==edge),None)
    if edge=='E_ORIGINAL':match=edges.get('legacy_edge')
    if edge.startswith('E5_'):
        match=next((x for x in read_json(project/'config/E5_EDGES.json')['edges'] if x['id']==edge),None)
        require(frame=='evaluation' and prep=='P1' and policies==['D0'],'E5 is descriptive P1/evaluation/D0 only')
    require(match is not None,'unregistered edge')
    sl={'pilot':'pilot_task_ids','evaluation':'evaluation_task_ids','external':'external_task_ids'}
    require(frame in sl,'frame not known')
    ids=plan[sl[frame]]
    for task in ids:
        mat_manifest=project/'runs/data'/str(task)/'MATRICES_MANIFEST.json'
        require(mat_manifest.is_file(),'missing matrix manifest '+str(task))
        m=read_json(mat_manifest)
        folds=sorted([(x['key'],str(mat_manifest.parent/x['file']),x['sha256']) for x in m['folds']
                      if x['preprocessing_id']==prep])
        require(len(folds)==30,'not exactly 30 frozen fold matrices')
        ds=next((x for x in plan['original_tasks']+plan['external_tasks'] if x['task_id']==task),None)
        require(ds is not None,'task missing from plan')
        seed_base=int(hashlib.sha256(('v3seed|'+str(task)).encode()).hexdigest()[:8],16)
        seeds={key:(seed_base+i*9973)%(2**31-1) for i,(key,_,_) in enumerate(folds)}
        if edge=='E_ORIGINAL':
            require(ds.get('legacy_seeds') is not None,'original seeds not recorded')
            seeds={key:int(ds['legacy_seeds'][key]) for key,_,_ in folds}
        for release in (match['old'],match['new']):
            for algo in (ALGORITHMS[:6] if edge=='E_ORIGINAL' or edge.startswith('E5_') else ALGORITHMS):
                for policy in policies:
                    jid='__'.join((edge,str(task),release,algo,policy,prep))
                    yield {'job_id':jid,'task_id':task,'release':release,'edge_id':edge,
                           'algo':algo,'policy':policy,'preprocessing_id':prep,'matrices':folds,
                           'seeds':seeds,'source_binding':source_binding(project,task),'legacy_e2':read_json(project/'config/E2_legacy_v26.json')['parameters'],
                           'new_settings':read_json(project/'config/D1_new_estimators_v302.json'),
                           'd2_effective':read_json(project/'config/D2_EFFECTIVE_OLD.json').get(edge,{}),
                           'out':str(result_dir/'results'/(jid+'.json')),'plan_digest':sha256(project/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json'),
                           'frozen_sha256':freeze_digest}

def verify_existing(j):
    p=Path(j['out']);require(p.is_file(),'result missing during resume')
    r=read_json(p)
    require(all(r.get(k)==j.get(k) for k in ('job_id','task_id','release','edge_id','algo','policy','preprocessing_id','plan_digest','frozen_sha256','environment_lock_sha256')) and r.get('job_spec_sha256')==digest_json(j),'old result job/hash mismatch')
    if r.get('status')=='SUPPORTED':
        z=p.with_suffix('.npz');require(z.exists() and sha256(z)==r.get('prediction_archive_sha256'),'prediction bytes mismatch')
    return r

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',default='.')
    parser.add_argument('--frame',choices=['pilot','evaluation','external'],required=True)
    parser.add_argument('--edge',required=True);parser.add_argument('--preprocessing',choices=['P1','P2'],required=True)
    parser.add_argument('--policies',default='D0,D1,D2');parser.add_argument('--out',required=True)
    parser.add_argument('--env-root',default=str(Path.home()/'versionshift_v3_envs'))
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--resume',action='store_true')
    a=parser.parse_args()
    require(1<=a.workers<=12,'invalid worker count')
    project=Path(a.project).resolve();out=Path(a.out).resolve();envroot=Path(a.env_root).resolve()
    freeze=project/'runs/plan/FROZEN_PROTOCOL.json'
    require(freeze.is_file(),'G2 frozen protocol absent; outcome runs prohibited')
    f=read_json(freeze);require(f.get('status')=='LOCALLY_FROZEN_NOT_EXTERNALLY_TIMESTAMPED','invalid protocol freeze')
    verify_frozen_sources(project, f)
    timestamp=project/'runs/plan/EXTERNAL_TIMESTAMP_RECORD.json'
    require(timestamp.is_file(),'G2 external timestamp record missing')
    t=read_json(timestamp)
    require(t.get('frozen_sha256')==sha256(freeze) and t.get('author_authorized') is True and t.get('public_url'),
            'external timestamp or author authorization missing')
    # Do not falsely assert timestamp verified independently; this checks author-supplied evidence only.
    allocation=read_json(project/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json')
    analysis,spec,analysis_hashes=load_settings(project,f['source_hashes'])
    release_names=[x['old'] for x in read_json(project/'config/edges.json')['new_primary_edges'] if x['id']==a.edge]
    release_names +=[x['new'] for x in read_json(project/'config/edges.json')['new_primary_edges'] if x['id']==a.edge]
    if a.edge=='E_ORIGINAL':
        orig=read_json(project/'config/edges.json')['legacy_edge']
        release_names=[orig['old'],orig['new']]
    if a.edge.startswith('E5_'):
        e=next((x for x in read_json(project/'config/E5_EDGES.json')['edges'] if x['id']==a.edge),None)
        require(e is not None,'E5 edge not in predeclared release sequence')
        release_names=[e['old'],e['new']]
    require(len(release_names)==2,'edge unresolved')
    versions={};env_lock_hashes={}
    for v in release_names:
        py=envroot/('sk-'+v)/'bin/python'
        require(py.exists(),'missing historical Python '+str(py))
        res=subprocess.run([str(py),'-I','-c','import sklearn; print(sklearn.__version__)'],capture_output=True,text=True)
        require(res.returncode==0 and res.stdout.strip()==v,'wrong sklearn in '+str(py))
        versions[v]=str(py)
        lockpaths=[envroot/('lock-'+v+'.txt'),envroot/('sk-'+v)/'FREEZE.txt',envroot/('oldstack_'+v.replace('.','')+'_FREEZE.txt')]
        chosen=next((q for q in lockpaths if q.is_file()),None)
        require(chosen is not None,'no environment package lock for '+v)
        env_lock_hashes[v]=sha256(chosen)
    if not out.exists():
        require(not a.resume,'cannot resume nonexistent run')
        (out/'results').mkdir(parents=True)
    elif not a.resume:raise SystemExit('STOP output exists; use --resume')
    lock=out/'SCHEDULER.lock'
    with lock.open('w') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('another scheduler holds this output lock')
        js=list(jobs_for_edge(project,allocation,a.edge,a.frame,a.preprocessing,
                              a.policies.split(','),versions,out,sha256(freeze)))
        for j in js:
            j['environment_lock_sha256']=env_lock_hashes[j['release']]
            frozen_source_rows(project,j['source_binding'],f)
        inv={'frozen_sha256':sha256(freeze),'selected_allocation_sha256':sha256(project/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json'),
             'job_ids_sha256':digest_json([x['job_id'] for x in js]),'jobs':len(js),
             'job_spec_digests':{j['job_id']:digest_json(j) for j in js},
             'analysis_hashes':analysis_hashes,
             'task_ids':allocation[{'pilot':'pilot_task_ids','evaluation':'evaluation_task_ids','external':'external_task_ids'}[a.frame]],
             'edge':a.edge,'frame':a.frame,'preprocessing':a.preprocessing,'policies':a.policies.split(','),
             'release_pythons':versions,'environment_lock_hashes':env_lock_hashes,'worker_sha256':sha256(project/'versionshift_v3/worker.py')}
        lockfile=out/'RUN_LOCK.json'
        if lockfile.exists():require(read_json(lockfile)==inv,'run-lock mismatch')
        else:write_once(lockfile,inv)
        remaining=[];skipped=0
        for j in js:
            p=Path(j['out'])
            if p.exists():verify_existing(j);skipped+=1
            else:remaining.append(j)
        print('RUN_LOCK_VERIFIED jobs',len(js),'skipped',skipped,'remaining',len(remaining),flush=True)
        logs=out/'job_logs';logs.mkdir(exist_ok=True)
        for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
            os.environ[v]='1'
        def launch(j):
            jid=j['job_id'];s=out/'job_specs';s.mkdir(exist_ok=True)
            spec=s/(jid+'.json')
            if spec.exists():require(read_json(spec)==j,'job spec changed')
            else:write_once(spec,j)
            try:
                proc=subprocess.run([versions[j['release']],'-I','-B',str(project/'versionshift_v3/worker.py'),str(spec)],
                                    capture_output=True,text=True,timeout=7200,env=os.environ.copy())
            except subprocess.TimeoutExpired as exc:
                write_once(j['out'],{'job_id':j['job_id'],'task_id':j['task_id'],
                    'release':j['release'],'edge_id':j['edge_id'],'algo':j['algo'],
                    'plan_digest':j['plan_digest'],'environment_lock_sha256':j['environment_lock_sha256'],
                    'job_spec_sha256':digest_json(j),
                    'policy':j['policy'],'preprocessing_id':j['preprocessing_id'],
                    'status':'TIMED_OUT','timeout_seconds':7200,'frozen_sha256':j['frozen_sha256'],
                    'worker_sha256':sha256(project/'versionshift_v3/worker.py')})
                return jid
            (logs/(jid+'.log')).write_text('returncode='+str(proc.returncode)+'\n'+proc.stdout+'\n'+proc.stderr[-2500:])
            if proc.returncode!=0:raise RuntimeError(jid+' worker process failed '+proc.stderr[-500:])
            verify_existing(j)
            return jid
        with futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            pending={pool.submit(launch,j):j for j in remaining}
            for i,fu in enumerate(futures.as_completed(pending),1):
                print('PROGRESS',skipped+i,'/',len(js),fu.result(),flush=True)
        print('RUN_COMPLETE',len(js),flush=True)
if __name__=='__main__':main()
