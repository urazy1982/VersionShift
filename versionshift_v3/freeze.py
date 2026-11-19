"""Strong prospective freeze. Does not permit fictitious hashes or empty evidence."""
import argparse,datetime,json,re,sys
from pathlib import Path
from .utils import read_json,sha256,write_once,require

def freeze(root,plan,output,author_approval):
    root=Path(root);p=read_json(plan)
    g0=root/'reports/G0_ARCHIVE_AUDIT.json'
    require(g0.is_file() and read_json(g0).get('status')=='PASS_G0_STRUCTURAL_ONLY','G0 archive audit has not passed')
    common=Path.home()/'versionshift_v3_envs/COMMON_STACK_CHECK.json'
    require(common.is_file() and read_json(common).get('status')=='PASS','G1 modern same-stack check not passed')
    require(p.get('status')=='PROPOSED_NOT_FROZEN','unexpected dataset plan status')
    require(len(p['pilot_task_ids'])==15 and len(p['evaluation_task_ids'])==48,'15/48 incomplete')
    require(len(p['external_task_ids'])==20,'20 external tasks not frozen')
    source=Path(root/'config/E2_legacy_v26.json');require(source.is_file(),'unverified v2.6 E2 file absent')
    e2=read_json(source)
    # Preserve exact E2 bytes; personal authorization lives in a separate record.
    from .author_gate import validate_e2_author_authorization
    e2_authorization=validate_e2_author_authorization(root,author_approval)
    original=root/'reports/G0_ARCHIVE_AUDIT.json'
    g=read_json(original)
    require(g.get('e2',{}).get('sha256')==sha256(source),'G0 E2 hash differs from author-reviewed map')
    origsha=g.get('required',{}).get('postreview/code/G3E/worker.py',{}).get('sha256')
    require(e2.get('source_sha256')==origsha,'E2 derived mapping source SHA does not match archived original')
    # The separate gate validates the actual signed approval and payload binding.
    from .historical_envs import validate_historical_readiness
    historical=validate_historical_readiness(root)
    vals={}
    required=['versionshift_v3/historical_envs.py','runs/plan/HISTORICAL_ENVIRONMENT_READINESS_v305.json','versionshift_v3/author_gate.py','runs/plan/E2_AUTHOR_AUTHORIZATION.json','runs/plan/AUTHOR_APPROVAL.txt','config/edges.json','config/E5_EDGES.json','config/analysis.json','config/E2_legacy_v26.json',
              'config/statistical_spec_v305.json','versionshift_v3/statistics.py','versionshift_v3/binding.py',
              'config/D1_new_estimators_v302.json','config/D2_EFFECTIVE_OLD.json','docs/PROTOCOL_v3.0.2_REVIEWED_DRAFT.md',
              'scripts/preflight_release.py','versionshift_v3/prepare.py',
              'versionshift_v3/validate_matrices.py','versionshift_v3/source_integrity.py',
              'scripts/export_raw_cc18.py','scripts/prepare_external_from_openml.py','scripts/prepare_all.py','scripts/prepare_index.py',
              'versionshift_v3/worker.py','versionshift_v3/audit.py',
              'versionshift_v3/plan.py','versionshift_v3/run.py','versionshift_v3/report.py',
              'scripts/aa_control.py','scripts/aggregate_primary.py','scripts/run_after_freeze.sh',
              'scripts/run_E5_descriptive.sh','scripts/build_original_fixedstack.sh',
              'docs/AMENDMENT_V305.md','scripts/v305/postcap.py',
              'scripts/v305/materialize_verify.py','scripts/v305/oldedge_feasibility.py',
              'scripts/v305/target_audit.py','scripts/v305/audit_index.py',
              'scripts/v305/solver_smoke.py']
    from .binding import load_settings,source_binding
    _,stat_spec,_=load_settings(root)
    require(stat_spec['author_review_status']=='AUTHOR_ACCEPTED','statistical definitions need actual author adjudication')
    for rel in required:
        f=root/rel;require(f.is_file(),'missing freeze file '+rel);vals[rel]=sha256(f)
    for key in ('legacy_plan_sha256','catalog_sha256'):
        require(re.fullmatch(r'[0-9a-f]{64}',p[key]) is not None,'bad input hash '+key)
    manifest=root/'runs/plan/MATRICES_INDEX.json'
    require(manifest.is_file(),'preprocessing matrix hashes not recorded')
    mi=read_json(manifest)
    require(mi.get('allocation_sha256')==sha256(plan),'matrix index does not match amended allocation')
    if p.get('design_amendment','').startswith('v3.0.5'):
        candidates=root/'runs/plan/EXTERNAL_CANDIDATES_POSTCAP_v305.csv'
        evidence=root/'runs/plan/POSTCAP_ALL_52_v305.json'
        require(candidates.is_file() and sha256(candidates)==p['catalog_sha256'],'post-cap catalog hash differs')
        require(evidence.is_file() and sha256(evidence)==p['postcap_evidence_sha256'],'post-cap universe evidence differs')
        vals['runs/plan/EXTERNAL_CANDIDATES_POSTCAP_v305.csv']=sha256(candidates)
        vals['runs/plan/POSTCAP_ALL_52_v305.json']=sha256(evidence)
        feasibility=root/'runs/plan/ORIGINAL_EDGE_COMMON_STACK_FEASIBILITY_v305.json'
        require(feasibility.is_file() and read_json(feasibility).get('status')=='PASS','original-edge common-stack feasibility not passed')
        vals['runs/plan/ORIGINAL_EDGE_COMMON_STACK_FEASIBILITY_v305.json']=sha256(feasibility)
    require(len(mi['task_manifests'])==83,'not all 63 core and 20 external tasks prepared')
    expected=set(p['pilot_task_ids']+p['evaluation_task_ids']+p['external_task_ids'])
    require({int(e['task_id']) for e in mi['task_manifests']}==expected,'matrix task IDs differ from prospective allocation')
    for entry in mi['task_manifests']:
        f=Path(entry['path']);require(f.is_file() and sha256(f)==entry['sha256'],'matrix manifest mismatch '+str(f))
        from .validate_matrices import validate_manifest
        validate_manifest(f, int(entry['task_id']))
        binding=source_binding(root,int(entry['task_id']))
        for key in ('matrix_manifest','labels','row_positions','provenance'):
            src=Path(binding[key+'_path']);vals[str(src.relative_to(root.resolve()))]=sha256(src)
    changelog=root/'runs/plan/CHANGELOG_ORACLE_16.csv'
    require(changelog.is_file(),'prospective 16-cell changelog frame missing')
    import csv
    with changelog.open(newline='') as handle: ch=list(csv.DictReader(handle))
    allowed={'change_expected','change_possible','fp_level_change_possible','no_change_expected'}
    keys={(r.get('edge_id'),r.get('estimator')) for r in ch}
    target={(edge,algo) for edge in ('E_LINEAR_DUAL','E_ADA_SAMME') for algo in ('LR','SVC','kNN','RF','GB','HGB','LinearSVC','AdaBoost')}
    require(len(ch)==16 and keys==target,'changelog not exactly 16 preregistered opportunities')
    require(all(r.get('category') in allowed and r.get('coder') and r.get('coded_on') and
                r.get('source_url','').startswith('https://') and r.get('trigger_condition') and
                r.get('coding_uncertainty') in ('low','medium','high') and
                'draft' not in r['coder'].lower() for r in ch),'changelog uncoded, missing evidence or draft')
    vals['runs/plan/MATRICES_INDEX.json']=sha256(manifest)
    vals['runs/plan/CHANGELOG_ORACLE_16.csv']=sha256(changelog)
    vals['selected_plan']=sha256(plan)
    vals['author_approval_sha256']=sha256(author_approval)
    vals['G0_ARCHIVE_AUDIT.json']=sha256(g0)
    vals['G1_COMMON_STACK_CHECK.json']=sha256(common)
    r={'status':'LOCALLY_FROZEN_NOT_EXTERNALLY_TIMESTAMPED','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'plan_sha256':sha256(plan),'source_hashes':vals,'e2_author_authorization':e2_authorization,'historical_environment_readiness':historical,'warning':'externally timestamp this SHA before running outcomes'}
    write_once(output,r);return r

def main():
    q=argparse.ArgumentParser();q.add_argument('--project',default='.')
    q.add_argument('--plan',required=True);q.add_argument('--author-approval',required=True)
    q.add_argument('--out',required=True);a=q.parse_args()
    r=freeze(a.project,a.plan,a.out,a.author_approval)
    print('LOCAL_FREEZE_CREATED',sha256(a.out),'EXTERNAL_TIMESTAMP_STILL_REQUIRED')
if __name__=='__main__':main()
