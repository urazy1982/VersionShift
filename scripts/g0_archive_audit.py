#!/usr/bin/env python3
"""Read-only structural inspection of a v2.6 checkout and an explicit E2 map.
Original outcome anchors from paper are documentation, not independently verified by this script.
"""
import argparse, json, hashlib, re, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from versionshift_v3.utils import sha256,write_once,read_json

NEEDED=["README.md","pipeline/run_versionshift_wsl.sh",
        "pipeline/addons/vs_fixedstack_verify.py","project/crossversion/controller.py",
        "project/crossversion/worker.py","results/data/PLAN.json",
        "postreview/code/G3E/worker.py","postreview/results/master_results.json"]
def audit(root, e2):
    report={'gate':'G0','status':'FAIL','repo_root':str(root.resolve()),'required':{},'missing':[],
            'e2':None,'task_frame':{},'observed_anchor_verification':'NOT_INDEPENDENTLY_REPLAYED'}
    for r in NEEDED:
        p=root/r
        if p.is_file():report['required'][r]={'sha256':sha256(p),'bytes':p.stat().st_size}
        else:report['missing'].append(r)
    if e2 and e2.is_file():
        report['e2']={'path':str(e2.resolve()),'sha256':sha256(e2)}
        try:
            obj=read_json(e2)
            vals=obj.get('parameters',obj)
            report['e2']['keys']=sorted(vals)
            if len(vals)!=6:report['missing'].append('E2 map must contain 6 estimator entries')
        except Exception as ex:report['missing'].append('invalid E2 JSON: '+repr(ex))
    else:report['missing'].append('explicit E2 map: supply original file or verified derived export')
    p=root/'results/data/PLAN.json'
    if p.is_file():
        try:
            plan=read_json(p)
            ds=plan.get('datasets',[])
            ids=[d.get('info',{}).get('task') for d in ds]
            report['task_frame']={'datasets':len(ds),'task_ids_present':all(isinstance(x,int) for x in ids),
                                  'unique_task_ids':len(set(ids))==len(ids),'plan_sha256':sha256(p),
                                  'task_ids':ids if len(ids)==63 else []}
            if len(ds)!=63 or not all(isinstance(x,int) for x in ids) or len(set(ids))!=63:
                report['missing'].append('63 unique original task IDs not proven')
        except Exception as ex:report['missing'].append('PLAN unreadable '+repr(ex))
    report['status']='PASS_G0_STRUCTURAL_ONLY' if not report['missing'] else 'FAIL'
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--v26-root',required=True);ap.add_argument('--e2-map',required=True)
    ap.add_argument('--out',required=True);a=ap.parse_args()
    r=audit(Path(a.v26_root),Path(a.e2_map));write_once(a.out,r)
    print(json.dumps({k:v for k,v in r.items() if k not in ('required','task_frame')},indent=2))
    if r['status']=='FAIL':raise SystemExit(2)
if __name__=='__main__':main()
