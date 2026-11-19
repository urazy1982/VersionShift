#!/usr/bin/env python3
"""Verify all P1/P2 matrices and create a single frozen index before G2."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from versionshift_v3.utils import read_json,sha256,write_once,require
from versionshift_v3.validate_matrices import validate_manifest
p=argparse.ArgumentParser();p.add_argument('--project',default='.');p.add_argument('--expected-count',type=int,default=83);a=p.parse_args()
root=Path(a.project)
plan=read_json(root/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json')
selected=plan['pilot_task_ids']+plan['evaluation_task_ids']+plan['external_task_ids']
require(len(selected)==a.expected_count and len(set(selected))==len(selected),'selected task count/uniqueness differs')
# Preserve nonselected historical materializations; index only the current proposal.
folders=[root/'runs/data'/str(tid)/'MATRICES_MANIFEST.json' for tid in sorted(selected)]
require(all(f.is_file() for f in folders),'selected matrix manifest missing')
require(len(folders)==a.expected_count,'matrix manifests count != expected '+str(a.expected_count))
rows=[]
for f in folders:
    m=validate_manifest(f)
    require(len(m['folds'])==60,'not 30 folds per P1/P2')
    for x in m['folds']:
        fp=f.parent/x['file'];require(fp.is_file() and sha256(fp)==x['sha256'],'matrix hash mismatch '+str(fp))
    rows.append({'task_id':m['task_id'],'path':str(f.resolve()),'sha256':sha256(f)})
require(len(set(x['task_id'] for x in rows))==len(rows),'duplicate prepared task')
plan=read_json(root/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json')
expected=set(plan['pilot_task_ids']+plan['evaluation_task_ids']+plan['external_task_ids'])
require({x['task_id'] for x in rows}==expected,'prepared IDs differ from selected plan')
write_once(root/'runs/plan/MATRICES_INDEX.json',{'task_manifests':rows,'allocation_sha256':sha256(root/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json')})
print('MATRIX_INDEX_READY',len(rows))
