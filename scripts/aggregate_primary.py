#!/usr/bin/env python3
"""Authenticate and replay exactly the predeclared two-edge D0/P1/eight family."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from versionshift_v3.utils import read_json,write_once,require
from versionshift_v3.binding import load_settings
from versionshift_v3.statistics import ALGOS,holm_two
from versionshift_v3.audit import audit_run

def validate_primary(report,edge,plan,hashes,analysis):
    lk=report['identity'];v=report.get('by_policy',{}).get('D0',{}).get('eight',{})
    require(lk['edge']==edge and lk['frame']=='evaluation' and lk['preprocessing']=='P1','primary run edge/frame/preprocessing differs')
    require(report['analysis_hashes']==hashes==lk['analysis_hashes'],'primary configuration binding differs')
    require(lk['task_ids']==plan['evaluation_task_ids'],'primary selected task IDs differ')
    require(v.get('panel')==ALGOS and v.get('K_complete')==analysis['primary_task_count']==48,'primary requires complete eight-estimator48-task panel')
    require(v.get('complete_task_ids')==sorted(plan['evaluation_task_ids']) and v.get('selected_task_ids')==sorted(plan['evaluation_task_ids']),'primary complete task IDs differ')
    return v

def aggregate_primary(paths,project):
    root=Path(project).resolve();freeze=read_json(root/'runs/plan/FROZEN_PROTOCOL.json');analysis,spec,hashes=load_settings(root,freeze['source_hashes']);plan=read_json(root/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json')
    require(set(paths)==set(spec['primary_family']),'primary family must contain precisely two new edges')
    records={};freeze_digest=None
    for edge in spec['primary_family']:
        supplied=read_json(paths[edge]);recomputed=audit_run(supplied['run_dir'],None,root)
        require(supplied==recomputed,'supplied audit differs from independent source-row replay')
        v=validate_primary(recomputed,edge,plan,hashes,analysis)
        fd=recomputed['identity']['frozen_sha256'];require(freeze_digest in (None,fd),'primary edges use different freezes');freeze_digest=fd
        p=v['p_one_sided_M_gt_margin'];require(p is not None or v['primary_test']['status'] in ('ZERO_VARIANCE_T_UNDEFINED','INSUFFICIENT_TASKS'),'unclassified absent primary test')
        records[edge]={'M_pp':v['M_pp'],'p_raw':p,'p_for_Holm':p if p is not None else 1.,
                       'test_status':v['primary_test']['status'],'conservative_untestable_handling':p is None,
                       'one_sided_lower95_pp':v['one_sided_lower95_pp'],'ci95_conditional':v['M_boot_ci95'],
                       'K_complete':v['K_complete'],'complete_task_ids':v['complete_task_ids'],
                       'winner_changes':v['winner_changes'],'affected':v['affected']}
    adjusted=holm_two({e:r['p_for_Holm'] for e,r in records.items()})
    for edge in records:records[edge]['p_holm']=adjusted[edge];records[edge]['reject_materiality_null']=records[edge]['p_raw'] is not None and adjusted[edge]<spec['primary_alpha']
    return {'status':'PRIMARY_48_TASK_HOLM' if all(r['p_raw'] is not None for r in records.values()) else 'PRIMARY_FAMILY_WITH_UNTESTABLE_EDGE',
            'evidence_scope':spec['evidence_scope'],'population_inference_scope':spec['population_inference_scope'],
            'family_size':2,'policy':'D0','frame':'evaluation','preprocessing':'P1','panel':ALGOS,'frozen_sha256':freeze_digest,'analysis_hashes':hashes,'edges':records,
            'caution':'Previously observed allocated task frame; undefined t tests use conservative1 for family adjustment, never fabricated raw p=0/1.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--linear',required=True);p.add_argument('--adaboost',required=True);p.add_argument('--out',required=True);p.add_argument('--project',default='.')
    a=p.parse_args();r=aggregate_primary({'E_LINEAR_DUAL':a.linear,'E_ADA_SAMME':a.adaboost},a.project);write_once(a.out,r);print(r['status'])
if __name__=='__main__':main()
