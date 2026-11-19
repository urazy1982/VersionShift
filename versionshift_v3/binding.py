"""Fail-closed bindings between configuration, selected tasks, jobs and raw rows."""
from pathlib import Path
from .utils import read_json,sha256,require,digest_json

CONFIG_FILES=('config/analysis.json','config/statistical_spec_v305.json')

def load_settings(root, expected_hashes=None):
    root=Path(root);hashes={p:sha256(root/p) for p in CONFIG_FILES}
    if expected_hashes is not None:
        require(all(expected_hashes.get(p)==h for p,h in hashes.items()),'analysis configuration hash mismatch')
    analysis=read_json(root/CONFIG_FILES[0]);spec=read_json(root/CONFIG_FILES[1])
    require(spec.get('evidence_scope')=='FINITE_ALLOCATED_TASK_FRAME' and spec.get('population_inference_scope')=='IID_TASK_MODEL_SENSITIVITY_ONLY','finite-frame/sensitivity scope changed')
    require(spec.get('E3_confirmatory_status')=='EXCLUDED_FUTURE_WORK_PROPOSAL_ONLY','E3 must remain excluded from confirmatory arms')
    require(analysis['primary_multiple_testing']=='Holm_across_two_new_edges_only','primary multiplicity changed')
    require(spec['primary_family']==['E_LINEAR_DUAL','E_ADA_SAMME'] and spec['primary_policy']=='D0' and
            spec['primary_frame']=='evaluation' and spec['primary_preprocessing']=='P1' and spec['primary_panel_size']==8,'primary family changed')
    require(analysis['repetitions']==6 and analysis['folds_per_repetition']==5,'fold design changed')
    require(analysis['primary_task_count']==48,'primary denominator changed')
    require(isinstance(analysis['bootstrap_samples'],int) and analysis['bootstrap_samples']>0,'bootstrap repetitions invalid')
    require(spec['primary_alpha']==.05 and spec['pair_ci_confidence']==.90 and spec['prevalence_confidence']==.95,'confidence levels differ from supplied protocol')
    require(spec['zero_variance_policy']=='UNDEFINED_T_INFERENCE_NO_EQUIVALENCE_HOLM_CONSERVATIVE_ONE','zero variance policy changed')
    require(spec['affected_rule'] in ('any_absolute_signed_pair_delta_gt_margin','M_gt_margin'),'unknown affected definition')
    require(spec['prevalence_model']=='one_sided_exact_binomial_CP_missing_tasks_counted_affected','unsupported prevalence model')
    require(spec['secondary_multiplicity']=='descriptive_no_confirmatory_secondary_claims','secondary family unspecified')
    require(analysis['primary_margin_pp']>0 and spec['winner_tie_tolerance_pp']>=0 and spec['material_reversal_margin_pp']>=0,'invalid practical margins')
    return analysis,spec,hashes

def source_binding(root,task_id):
    root=Path(root);mf=root/f'runs/data/{task_id}/MATRICES_MANIFEST.json'
    core=root/f'runs/raw/{task_id}'
    if (core/'labels.npy').is_file():
        labels=core/'labels.npy';positions=core/'original_raw_row_positions.npy';provenance=core/'RAW_PROVENANCE.json'
    else:
        raw=root/f'reports/V305_20261117T091900Z/candidates/{task_id}'
        labels=raw/'labels.npy';positions=raw/'row_positions.npy';provenance=raw/'POSTCAP.json'
    require(labels.is_file() and positions.is_file() and mf.is_file() and provenance.is_file(),'source row provenance missing')
    m=read_json(mf);pr=read_json(provenance)
    require(m['task_id']==task_id,'manifest source task differs')
    want_labels=m.get('raw_labels_sha256',pr.get('files_sha256',{}).get('labels.npy'))
    require(want_labels==sha256(labels),'raw label provenance hash differs')
    if m.get('external_source'):
        require(pr['task_id']==task_id and pr['dataset_id']==m['dataset_id'] and pr['dataset_version']==m['dataset_version'],'candidate source revision differs')
        require(pr['source']['arff_sha256']==m['source_integrity']['arff_sha256'],'candidate ARFF differs from manifest')
    want_positions=pr.get('indices_sha256',pr.get('files_sha256',{}).get('row_positions.npy'))
    require(want_positions==sha256(positions),'raw row position provenance differs')
    if 'raw_positions_sha256' in m:require(m['raw_positions_sha256']==sha256(positions),'manifest raw positions differ')
    return {'provenance_path':str(provenance.resolve()),'provenance_sha256':sha256(provenance),'matrix_manifest_path':str(mf.resolve()),'matrix_manifest_sha256':sha256(mf),
            'labels_path':str(labels.resolve()),'labels_sha256':sha256(labels),
            'row_positions_path':str(positions.resolve()),'row_positions_sha256':sha256(positions)}

def frozen_source_rows(root,binding,frozen):
    root=Path(root).resolve()
    for key in ('matrix_manifest','labels','row_positions','provenance'):
        p=Path(binding[key+'_path']).resolve();require(p.is_relative_to(root),'source path outside project')
        require(sha256(p)==binding[key+'_sha256'],'source binding bytes changed')
        require(frozen['source_hashes'].get(str(p.relative_to(root)))==sha256(p),'source rows not bound by freeze')
    return True

def identity_fields(job):
    return {k:job[k] for k in ('job_id','task_id','release','edge_id','algo','policy','preprocessing_id','plan_digest','frozen_sha256','environment_lock_sha256')}
