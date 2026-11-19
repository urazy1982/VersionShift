"""Artificial row/prediction archives only. Never fits study tasks or creates approval."""
import json,shutil
from pathlib import Path
import numpy as np
from versionshift_v3.utils import sha256,digest_json
from versionshift_v3.run import jobs_for_edge
from versionshift_v3.binding import load_settings,source_binding
from versionshift_v3.statistics import ALGOS
ROOT=Path(__file__).resolve().parents[1]
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))

def synthetic_run(tmp_path,policies=('D0',),n_tasks=1):
    root=tmp_path/'synthetic_project';run=root/'synthetic_run';(run/'results').mkdir(parents=True)
    for name in ('analysis.json','statistical_spec_v305.json','edges.json','E2_legacy_v26.json','D1_new_estimators_v302.json','D2_EFFECTIVE_OLD.json'):
        p=root/'config'/name;p.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'config'/name,p)
    p=root/'versionshift_v3/worker.py';p.parent.mkdir();shutil.copy2(ROOT/'versionshift_v3/worker.py',p)
    tasks=[-101-i for i in range(n_tasks)];keys=[f'r{r}_f{f}' for r in range(6) for f in range(5)]
    plan={'status':'SYNTHETIC_ENGINEERING_ONLY','pilot_task_ids':[],'evaluation_task_ids':tasks,'external_task_ids':[],
          'original_tasks':[{'task_id':t,'legacy_seeds':{k:42 for k in keys}} for t in tasks],'external_tasks':[]}
    put(root/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json',plan)
    for task in tasks:
        raw=root/f'runs/raw/{task}';raw.mkdir(parents=True);y=np.arange(10)%2;np.save(raw/'labels.npy',y);np.save(raw/'original_raw_row_positions.npy',np.arange(10)+1000)
        folder=root/f'runs/data/{task}';folder.mkdir(parents=True);entries=[]
        for key in keys:
            f=int(key[-1]);te=np.arange(f*2,f*2+2);tr=np.setdiff1d(np.arange(10),te)
            for prep in ('P1','P2'):
                path=folder/f'{key}_{prep}.npz';np.savez_compressed(path,X_train=np.zeros((8,2)),X_test=np.zeros((2,2)),y_train=y[tr],y_test=y[te],train_indices=tr,test_indices=te)
                entries.append({'key':key,'preprocessing_id':prep,'file':path.name,'sha256':sha256(path),'n_features':2})
        put(raw/'RAW_PROVENANCE.json',{'indices_sha256':sha256(raw/'original_raw_row_positions.npy')})
        put(folder/'MATRICES_MANIFEST.json',{'task_id':task,'folds':entries,'raw_labels_sha256':sha256(raw/'labels.npy'),'raw_positions_sha256':sha256(raw/'original_raw_row_positions.npy')})
    _,_,hashes=load_settings(root)
    inventory={**hashes,'selected_plan':sha256(root/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json'),'versionshift_v3/worker.py':sha256(root/'versionshift_v3/worker.py')}
    for task in tasks:
        sb=source_binding(root,task)
        for k in ('matrix_manifest','labels','row_positions','provenance'):inventory[str(Path(sb[k+'_path']).relative_to(root))]=sb[k+'_sha256']
    # Explicitly synthetic integrity fixture, never a genuine prospective freeze.
    put(root/'runs/plan/FROZEN_PROTOCOL.json',{'status':'SYNTHETIC_ENGINEERING_FIXTURE_NOT_A_FREEZE','source_hashes':inventory,'plan_sha256':sha256(root/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json')})
    fd=sha256(root/'runs/plan/FROZEN_PROTOCOL.json');versions={'1.4.2':'SYNTHETIC_NO_INTERPRETER','1.5.2':'SYNTHETIC_NO_INTERPRETER'};envs={v:'synthetic_environment_hash' for v in versions}
    jobs=list(jobs_for_edge(root,plan,'E_LINEAR_DUAL','evaluation','P1',list(policies),versions,run,fd))
    for j in jobs:j['environment_lock_sha256']=envs[j['release']]
    lock={'frozen_sha256':fd,'selected_allocation_sha256':sha256(root/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json'),
          'job_ids_sha256':digest_json([j['job_id'] for j in jobs]),'job_spec_digests':{j['job_id']:digest_json(j) for j in jobs},
          'jobs':len(jobs),'edge':'E_LINEAR_DUAL','frame':'evaluation','preprocessing':'P1','policies':list(policies),
          'release_pythons':versions,'environment_lock_hashes':envs,'worker_sha256':sha256(root/'versionshift_v3/worker.py'),'task_ids':tasks,'analysis_hashes':hashes}
    put(run/'RUN_LOCK.json',lock)
    for j in jobs:
        unsupported=j['algo']=='HGB' and j['policy']=='D1'
        rec={k:j[k] for k in ('job_id','task_id','release','edge_id','algo','policy','preprocessing_id','plan_digest','frozen_sha256','environment_lock_sha256')}
        rec.update(worker_sha256=lock['worker_sha256'],job_spec_sha256=digest_json(j),source_binding=j['source_binding'],status='UNSUPPORTED' if unsupported else 'SUPPORTED')
        if unsupported:rec['error']='synthetic exact E2 HGB static nonportability'
        else:
            arrays={};scores=[]
            for key,path,h in j['matrices']:
                with np.load(path) as z:
                    y=z['y_test'];pred=y.copy()
                    if j['release']=='1.5.2' and j['algo']=='RF':pred=1-y
                    arrays[key+'_y']=y;arrays[key+'_pred']=pred;arrays[key+'_test_indices']=z['test_indices'];arrays[key+'_source_row_ids']=z['test_indices']+1000
                    scores.append(100.*float(np.mean(pred==y)))
            path=Path(j['out']).with_suffix('.npz');np.savez_compressed(path,**arrays);rec.update(prediction_archive_sha256=sha256(path),balanced_accuracy_pp=scores,fold_keys=keys,warning_count=0,n_iter=[[]]*30)
        put(Path(j['out']),rec)
    return root,run,jobs

def add_synthetic_edge(root,edge='E_ADA_SAMME'):
    """Second artificial edge on the exact same synthetic integrity fixture."""
    root=Path(root);run=root/'synthetic_second_edge';(run/'results').mkdir(parents=True)
    plan=json.loads((root/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json').read_text());_,_,hashes=load_settings(root)
    fd=sha256(root/'runs/plan/FROZEN_PROTOCOL.json');versions={'1.5.2':'SYNTHETIC_NO_INTERPRETER','1.6.1':'SYNTHETIC_NO_INTERPRETER'};envs={v:'synthetic_environment_hash' for v in versions}
    jobs=list(jobs_for_edge(root,plan,edge,'evaluation','P1',['D0'],versions,run,fd))
    for j in jobs:j['environment_lock_sha256']=envs[j['release']]
    lock={'frozen_sha256':fd,'selected_allocation_sha256':sha256(root/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json'),
          'job_ids_sha256':digest_json([j['job_id'] for j in jobs]),'job_spec_digests':{j['job_id']:digest_json(j) for j in jobs},
          'jobs':len(jobs),'edge':edge,'frame':'evaluation','preprocessing':'P1','policies':['D0'],
          'release_pythons':versions,'environment_lock_hashes':envs,'worker_sha256':sha256(root/'versionshift_v3/worker.py'),'task_ids':plan['evaluation_task_ids'],'analysis_hashes':hashes}
    put(run/'RUN_LOCK.json',lock)
    for j in jobs:
        rec={k:j[k] for k in ('job_id','task_id','release','edge_id','algo','policy','preprocessing_id','plan_digest','frozen_sha256','environment_lock_sha256')}
        rec.update(worker_sha256=lock['worker_sha256'],job_spec_sha256=digest_json(j),source_binding=j['source_binding'],status='SUPPORTED')
        arrays={};scores=[]
        for key,path,h in j['matrices']:
            with np.load(path) as z:
                y=z['y_test'];pred=y.copy()
                if j['release']=='1.6.1' and j['algo']=='RF':pred=1-y
                arrays[key+'_y']=y;arrays[key+'_pred']=pred;arrays[key+'_test_indices']=z['test_indices'];arrays[key+'_source_row_ids']=z['test_indices']+1000;scores.append(100.*float(np.mean(pred==y)))
        p=Path(j['out']).with_suffix('.npz');np.savez_compressed(p,**arrays);rec.update(prediction_archive_sha256=sha256(p),balanced_accuracy_pp=scores,fold_keys=[key for key,_,_ in j['matrices']]);put(Path(j['out']),rec)
    return run
