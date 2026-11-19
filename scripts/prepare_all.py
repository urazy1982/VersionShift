#!/usr/bin/env python3
"""Resumable *pre-outcome* preparation on Ubuntu-Semantic; no model fits.
Checks all existing task manifests instead of overwriting them.
"""
import argparse,sys,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from versionshift_v3.utils import read_json,sha256,require
from versionshift_v3.validate_matrices import validate_manifest
from scripts.export_raw_cc18 import reconstruct
from versionshift_v3.prepare import materialize_raw_task
from scripts.prepare_external_from_openml import go as external_go
p=argparse.ArgumentParser();p.add_argument('--project',default='.')
p.add_argument('--legacy-plan-dir',required=True);a=p.parse_args()
root=Path(a.project).resolve()
lock=(root/'runs/prepare.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('STOP: another preparation controller holds runs/prepare.lock')
require(not (root/'runs/plan/FROZEN_PROTOCOL.json').exists(),'already frozen')
plan=read_json(root/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json')
require(sha256(Path(a.legacy_plan_dir)/'PLAN.json')==plan['legacy_plan_sha256'],'legacy PLAN not installed or changed')
for i,item in enumerate(plan['original_tasks'],1):
    tid=item['task_id'];raw=root/'runs/raw'/str(tid);out=root/'runs/data'/str(tid)
    if not raw.exists():reconstruct(tid,a.legacy_plan_dir,raw)
    if not out.exists():
        cs=read_json(raw/'columns.json')
        materialize_raw_task(tid,a.legacy_plan_dir,raw/'X.csv',raw/'labels.npy',
            cs['numeric_columns'],cs['categorical_columns'],out)
    require((out/'MATRICES_MANIFEST.json').exists(),'incomplete core task '+str(tid))
    validate_manifest(out/'MATRICES_MANIFEST.json',tid)
    print('CORE',i,'/63',tid,flush=True)
for i,tid in enumerate(plan['external_task_ids'],1):
    out=root/'runs/data'/str(tid)
    if not out.exists():external_go(tid,root,out)
    require((out/'MATRICES_MANIFEST.json').exists(),'incomplete external task '+str(tid))
    validate_manifest(out/'MATRICES_MANIFEST.json',tid)
    print('EXTERNAL',i,'/20',tid,flush=True)
print('DATASET_PREPARATION_DONE',len(plan['original_tasks'])+len(plan['external_task_ids']))
