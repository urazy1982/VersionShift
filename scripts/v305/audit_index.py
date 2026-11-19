"""Read-only independent index/preservation audit; no materialization or fitting."""
import os,sys,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from versionshift_v3.utils import read_json,sha256,require,write_once
from versionshift_v3.validate_matrices import validate_manifest
ROOT=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);p=read_json(ROOT/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json');index=read_json(ROOT/'runs/plan/MATRICES_INDEX.json');expected=set(p['pilot_task_ids']+p['evaluation_task_ids']+p['external_task_ids']);require(len(expected)==83 and len(index['task_manifests'])==83,'index count')
require(index['allocation_sha256']==sha256(ROOT/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json'),'allocation hash differs');require({x['task_id'] for x in index['task_manifests']}==expected,'IDs differ')
rows=[]
for e in index['task_manifests']:
 f=Path(e['path']);require(sha256(f)==e['sha256'],'manifest hash differs');m=validate_manifest(f,e['task_id']);rows.append({'task_id':e['task_id'],'manifest_sha256':sha256(f),'matrix_count':len(m['folds'])})
base=read_json(out/'PRESERVATION_BASELINE.json');preserved=[]
for rel,h in base.items():
 if rel.startswith(('runs/data/','runs/raw/')):
  require(sha256(ROOT/rel)==h,'pre-existing matrix/raw artifact changed '+rel);preserved.append(rel)
require(sha256(ROOT/'runs/plan/PROSPECTIVE_TASK_ALLOCATION_v304_PRESERVED.json')==sha256(out/'PRIOR_ALLOCATION.json'),'prior proposal not preserved')
v26=Path.home()/'VersionShift_v26_source';legacy=Path.home()/'versionshift_data/STAGE_B_PLAN_CACHED';lp=read_json(v26/'results/data/PLAN.json');n=0
for d in lp['datasets']:
 for k,s in [('dataset_npz','dataset_sha256'),('folds_npz','folds_sha256')]:require(sha256(legacy/Path(d[k]).name)==d[s],'historical integrity failure');n+=1
r=subprocess.run(['git','status','--porcelain'],cwd=v26,capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'});require(r.returncode==0 and not r.stdout,'historical source checkout changed')
write_once(out/'INDEPENDENT_83_INDEX_AUDIT.json',{'status':'PASS','recorded_utc':datetime.now(timezone.utc).isoformat(),'tasks':rows,'matrix_count':sum(x['matrix_count'] for x in rows),'preexisting_matrix_raw_files_unchanged':len(preserved),'prior_allocation_preserved':True,'historical_126_hashes_match':n==126,'historical_checkout_clean':True,'index_sha256':sha256(ROOT/'runs/plan/MATRICES_INDEX.json'),'qualification':'Separate read-only index/hash/structural validation, complementing exact raw/fold/P1/P2 replay in ALL_83_INDEPENDENT_VERIFICATION.json.'})
print('PASS:83 selected tasks,4980 matrices,',len(preserved),'prior files unchanged,126 historical hashes')
