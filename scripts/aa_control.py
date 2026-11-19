#!/usr/bin/env python3
"""Same-version repeat-fit A/A control on frozen P1 fold before opening new edge comparisons."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from versionshift_v3.utils import read_json,sha256,write_once,require

PROBE='''
import json,sys,os,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,sys.argv[1]);from versionshift_v3.worker import init_model
cfg=json.loads(Path(sys.argv[2]).read_text());z=np.load(cfg['matrix'],allow_pickle=False)
answers={}
for algo in cfg['algorithms']:
    for policy in cfg['policies']:
        preds=[];statuses=[]
        for _ in range(2):
            m,status,reason=init_model(algo,policy,42,cfg['release'],cfg['e2'],cfg['new'])
            if status=='SUPPORTED':
                try:m.fit(z['X_train'],z['y_train']);preds.append(m.predict(z['X_test']).astype(int).tolist())
                except Exception as ex:status='FIT_FAILED';reason=repr(ex)
            statuses.append(status)
        answers[algo+':'+policy]={'status':statuses,'pred_equal':preds[0]==preds[1] if len(preds)==2 else None}
print(json.dumps(answers))
'''
def main():
    p=argparse.ArgumentParser();p.add_argument('--project',default='.')
    p.add_argument('--env-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.project).resolve();plan=read_json(root/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json')
    task=plan['pilot_task_ids'][0]
    m=read_json(root/'runs/data'/str(task)/'MATRICES_MANIFEST.json')
    folder=root/'runs/data'/str(task)
    entry=next(x for x in m['folds'] if x['key']=='r0_f0' and x['preprocessing_id']=='P1')
    matrix=folder/entry['file'];require(sha256(matrix)==entry['sha256'],'A/A input matrix differs')
    e2=read_json(root/'config/E2_legacy_v26.json')['parameters'];new=read_json(root/'config/D1_new_estimators_v302.json')
    statuses={}
    for v in ['1.4.2','1.5.2','1.6.1']:
        py=Path(a.env_root)/('sk-'+v)/'bin/python';require(py.exists(),'missing AA interpreter '+str(py))
        spec=root/'runs/aa_specs'/('AA_PROBE_'+v+'.json')
        spec.parent.mkdir(parents=True,exist_ok=True)
        expected={'matrix':str(matrix),'algorithms':['LR','SVC','kNN','RF','GB','HGB','LinearSVC','AdaBoost'],
                  'policies':['D0','D1'],'release':v,'e2':e2,'new':new}
        if spec.exists():require(read_json(spec)==expected,'existing A/A probe specification changed')
        else:spec.write_text(json.dumps(expected,sort_keys=True))
        cmd=[str(py),'-I','-c',PROBE,str(root),str(spec)]
        env=os.environ.copy();env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
        sub=subprocess.run(cmd,capture_output=True,text=True,timeout=1800,env=env)
        require(sub.returncode==0,'AA worker failed '+sub.stderr[-500:])
        result=json.loads(sub.stdout.strip().splitlines()[-1])
        require(all(vv['pred_equal'] is True for vv in result.values() if vv['status']==['SUPPORTED','SUPPORTED']),
                'AA prediction drift')
        statuses[v]=result
    write_once(a.out,{'status':'AA_SAME_VERSION_PASS','fixture_task':task,'matrix_sha256':sha256(matrix),'versions':statuses})
    print('AA_SAME_VERSION_PASS',task)
if __name__=='__main__':main()
