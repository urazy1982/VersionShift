"""Independent numerical oracles and adversarial synthetic integrity fixtures."""
import copy,json,itertools
from pathlib import Path
import numpy as np
import pytest
from scipy import stats
from versionshift_v3.statistics import ALGOS,evaluate_edge,t_inference,prevalence_bound,holm_two,paired_mitigation
from versionshift_v3.audit import audit_run,score_replay
from versionshift_v3.utils import sha256
from synthetic_statistics_fixtures import synthetic_run,put

def records(n=48,scale=.01):
    return {(t,v,a):65+i+(scale*(t+1)*(i+1) if v=='new' else 0) for t in range(n) for v in ('old','new') for i,a in enumerate(ALGOS)}

def test_independent_M_signed_pairs_bootstrap_and_t_bounds():
    data=records();r=evaluate_edge(data,expected_tasks=list(range(48)),releases=['old','new'],bootstrap_reps=200,seed=19)
    # Separate direct definition uses all four score terms, not production dx.
    values=[]
    for t in range(48):values.append(np.mean([abs((data[t,'new',a]-data[t,'new',b])-(data[t,'old',a]-data[t,'old',b])) for a,b in itertools.combinations(ALGOS,2)]))
    m=np.mean(values);se=np.std(values,ddof=1)/np.sqrt(48)
    assert r['M_pp']==pytest.approx(m)
    assert r['one_sided_lower95_pp']==pytest.approx(m-stats.t.ppf(.95,47)*se)
    assert r['p_one_sided_M_gt_margin']==pytest.approx(stats.t.sf((m-.5)/se,47))
    draws=np.array(values)[np.random.RandomState(19).randint(0,48,(200,48))].mean(axis=1)
    assert r['M_boot_ci95']==pytest.approx(np.percentile(draws,[2.5,97.5]))
    a,b=ALGOS[:2];d=np.array([(data[t,'new',a]-data[t,'new',b])-(data[t,'old',a]-data[t,'old',b]) for t in range(48)])
    ci=[d.mean()-stats.t.ppf(.95,47)*d.std(ddof=1)/np.sqrt(48),d.mean()+stats.t.ppf(.95,47)*d.std(ddof=1)/np.sqrt(48)]
    assert r['signed_pair_intervals'][0]['ci90_pp']==pytest.approx(ci)
    for row in r['algorithm_leverage']:assert sum(row['contribution_to_M_pp'].values())==pytest.approx(values[row['task_id']])

@pytest.mark.parametrize('x',[[0]*48,[.1]*48,[.5]*48,[1.]*48])
def test_zero_variance_never_fabricates_t_p_or_equivalence(x):
    r=t_inference(x);assert r['status']=='ZERO_VARIANCE_T_UNDEFINED';assert r['p_one_sided'] is None and r['one_sided_lower_pp'] is None and r['ci_two_sided_pp'] is None

def test_insufficient_nonfinite_and_prevalence_boundaries():
    assert t_inference([.2])['status']=='INSUFFICIENT_TASKS'
    with pytest.raises(RuntimeError):t_inference([1,np.nan])
    assert prevalence_bound(0,48)['upper']==pytest.approx(1-.05**(1/48))
    assert prevalence_bound(48,48)['upper']==1
    assert prevalence_bound(0,0)['upper'] is None
    with pytest.raises(RuntimeError):prevalence_bound(49,48)

def test_equivalence_requires_every_pair_share_bound_and_assumptions():
    data=records(scale=.00001);r=evaluate_edge(data,releases=['old','new'],bootstrap_reps=10)
    assert r['equivalence']['all_signed_pairs_within_margin'];assert r['equivalence']['share_bound_below_threshold'];assert r['equivalence']['status']=='INCONCLUSIVE'
    r=evaluate_edge(data,releases=['old','new'],bootstrap_reps=10,spec={'prevalence_model_justified':True})
    assert r['equivalence']['status']=='EQUIVALENT_CONDITIONAL_ON_TASK_MODEL'
    r=evaluate_edge(data,releases=['old','new'],expected_tasks=list(range(50)),bootstrap_reps=10,spec={'prevalence_model_justified':True})
    assert r['affected_share_upper_bound']['affected']==2 and r['affected_share_upper_bound']['denominator']==50
    assert r['equivalence']['status']=='INCONCLUSIVE' and r['K_incomplete']==2

def test_complete_missing_task_is_counted_no_zero_imputation():
    r=evaluate_edge(records(3),expected_tasks=list(range(5)),releases=['old','new'],bootstrap_reps=10)
    assert r['incomplete_tasks']==[3,4] and r['selected_denominator']==5 and r['task_failure_fraction']==.4

def test_affected_definitions_are_separate_and_ties_margin_winners_reversals():
    data={(t,v,a):50. for t in range(3) for v in ('old','new') for a in ALGOS}
    for a in ALGOS:data[0,'old',a]=50.;data[0,'new',a]=50.
    data[0,'old','LR']=51.;data[0,'new','SVC']=51.
    data[1,'old','LR']=50.25;data[1,'new','SVC']=50.25
    # isolated shift .6 yields any pair material, M only .15
    data[2,'new','RF']=50.6
    r=evaluate_edge(data,releases=['old','new'],bootstrap_reps=10)
    assert r['margin_qualified_winners']['0.5']['eligible_denominator']==1
    assert r['margin_qualified_winners']['0.5']['winner_changes']==1
    assert r['margin_qualified_winners']['0.25']['eligible_denominator']==1 # equality excluded
    assert r['margin_qualified_winners']['0.0']['eligible_denominator']==2 # tied old task2 excluded
    assert r['per_task'][2]['affected_any_pair'] and not r['per_task'][2]['affected_M_gt_margin']
    assert r['material_reversals'][0]['count']==1
    assert r['material_reversal_summary']['material_gap_eligible_pairs']==1 and r['material_reversal_summary']['complete_task_pair_opportunities']==84
    assert r['per_estimator_shifts']['RF']['mean_shift_pp']==pytest.approx(.2)

def test_holm_ordering_ties_and_invalid_values():
    assert holm_two({'a':.04,'b':.01})=={'b':.02,'a':.04}
    assert holm_two({'a':.9,'b':.9})=={'a':1.,'b':1.}
    assert holm_two({'a':1.,'b':.01})=={'b':.02,'a':1.}
    with pytest.raises(RuntimeError):holm_two({'a':float('nan'),'b':.1})

def test_paired_denominators_missing_and_zero_attenuation_denominator():
    a={'selected_task_ids':[1,2,3],'per_task':[{'task_id':1,'M_pp':0.},{'task_id':2,'M_pp':2.}]};b={'selected_task_ids':[1,2,3],'per_task':[{'task_id':1,'M_pp':0.}]}
    r=paired_mitigation(a,b,bootstrap_reps=10);assert r['paired_tasks']==1 and r['selected_denominator']==3 and r['relative_attenuation'] is None and r['bootstrap_zero_denominator_count']==10
    r=paired_mitigation(a,{'selected_task_ids':[1,2,3],'per_task':[]});assert r['status']=='NOT_IDENTIFIABLE' and r['paired_failure_fraction']==1

def test_end_to_end_source_replay_and_nonportable_HGB(tmp_path):
    root,run,jobs=synthetic_run(tmp_path,('D0','D1'));r=audit_run(run,tmp_path/'audit.json',root)
    assert r['status']=='RAW_SOURCE_ROW_REPLAY_PASS' and r['status_counts']['SUPPORTED']==30 and r['status_counts']['UNSUPPORTED']==2
    assert r['portability_by_policy']['D1']['nonportable_jobs']==2
    assert r['by_policy']['D1']['legacy_six']['K_complete']==0 and r['mitigation']['eight']['paired_tasks']==0
    rf=next(x for x in r['prediction_disagreement'] if x['estimator']=='RF' and x['policy']=='D0')
    assert rf['pooled_disagreement_rate']==1 and rf['prediction_occurrences']==60

@pytest.mark.parametrize('field,value',[('edge','E_ADA_SAMME'),('frame','pilot'),('preprocessing','P2'),('task_ids',[999])])
def test_run_lock_identity_attacks_rejected(tmp_path,field,value):
    root,run,jobs=synthetic_run(tmp_path);p=run/'RUN_LOCK.json';x=json.loads(p.read_text());x[field]=value;put(p,x)
    with pytest.raises((RuntimeError,KeyError)):audit_run(run,None,root)

def test_missing_whole_job_explicit_and_duplicate_unsupported_rejected(tmp_path):
    root,run,jobs=synthetic_run(tmp_path,('D0','D1'));Path(jobs[0]['out']).unlink();r=audit_run(run,None,root)
    assert r['status_counts']['MISSING_JOB']==1 and r['by_policy']['D0']['eight']['K_complete']==0
    p=next(Path(j['out']) for j in jobs if j['algo']=='HGB' and j['policy']=='D1');(run/'results'/'duplicate.json').write_bytes(p.read_bytes())
    with pytest.raises(RuntimeError):audit_run(run,None,root)

@pytest.mark.parametrize('suffix',['_test_indices','_source_row_ids','_y'])
def test_semantic_row_tamper_with_rehashed_archive_rejected(tmp_path,suffix):
    root,run,jobs=synthetic_run(tmp_path);p=Path(jobs[0]['out']);z=p.with_suffix('.npz')
    with np.load(z) as a:arrays={k:a[k].copy() for k in a.files}
    arrays['r0_f0'+suffix]=arrays['r0_f0'+suffix][::-1];np.savez_compressed(z,**arrays)
    r=json.loads(p.read_text());r['prediction_archive_sha256']=sha256(z);put(p,r)
    with pytest.raises(RuntimeError):audit_run(run,None,root)

def test_frozen_configuration_tamper_rejected(tmp_path):
    root,run,jobs=synthetic_run(tmp_path);p=root/'config/analysis.json';x=json.loads(p.read_text());x['bootstrap_seed']=42;put(p,x)
    with pytest.raises(RuntimeError,match='frozen evidence changed'):audit_run(run,None,root)

def test_primary_identity_validates_actual_ids_and_panel_and_family():
    from scripts.aggregate_primary import validate_primary
    from versionshift_v3.binding import load_settings
    root=Path(__file__).resolve().parents[1];analysis,spec,hashes=load_settings(root);ids=list(range(48));plan={'evaluation_task_ids':ids}
    r={'identity':{'edge':'E_LINEAR_DUAL','frame':'evaluation','preprocessing':'P1','task_ids':ids,'analysis_hashes':hashes},'analysis_hashes':hashes,'by_policy':{'D0':{'eight':{'panel':ALGOS,'K_complete':48,'complete_task_ids':ids,'selected_task_ids':ids}}}}
    assert validate_primary(r,'E_LINEAR_DUAL',plan,hashes,analysis)['K_complete']==48
    for field,value in [('edge','E_ADA_SAMME'),('preprocessing','P2'),('frame','pilot'),('task_ids',list(range(1,49)))]:
        bad=copy.deepcopy(r);bad['identity'][field]=value
        with pytest.raises(RuntimeError):validate_primary(bad,'E_LINEAR_DUAL',plan,hashes,analysis)
    bad=copy.deepcopy(r);bad['by_policy']['D0']['eight']['panel']=ALGOS[:6]
    with pytest.raises(RuntimeError):validate_primary(bad,'E_LINEAR_DUAL',plan,hashes,analysis)

def test_legacy_external_manifest_uses_verified_candidate_provenance(tmp_path):
    from versionshift_v3.binding import source_binding
    task=-500;raw=tmp_path/f'reports/V305_20261117T091900Z/candidates/{task}';raw.mkdir(parents=True)
    np.save(raw/'labels.npy',np.array([0,1]));np.save(raw/'row_positions.npy',np.array([4,8]));mf=tmp_path/f'runs/data/{task}/MATRICES_MANIFEST.json'
    put(mf,{'task_id':task,'external_source':True,'dataset_id':123,'dataset_version':1,'source_integrity':{'arff_sha256':'SYNTHETIC'}})
    pr={'task_id':task,'dataset_id':123,'dataset_version':1,'source':{'arff_sha256':'SYNTHETIC'},'files_sha256':{'labels.npy':sha256(raw/'labels.npy'),'row_positions.npy':sha256(raw/'row_positions.npy')}}
    put(raw/'POSTCAP.json',pr);before=sha256(mf);b=source_binding(tmp_path,task);assert b['labels_sha256']==pr['files_sha256']['labels.npy'] and sha256(mf)==before
    pr['dataset_version']=2;put(raw/'POSTCAP.json',pr)
    with pytest.raises(RuntimeError,match='revision differs'):source_binding(tmp_path,task)

@pytest.mark.parametrize('status',['UNSUPPORTED','REMOVED_BEHAVIOR','INEFFECTIVE','API_REMOVED','FIT_FAILED','TIMED_OUT','MISSING_JOB'])
def test_every_failure_state_has_explicit_job_denominator(tmp_path,status):
    root,run,jobs=synthetic_run(tmp_path);p=Path(jobs[0]['out']);r=json.loads(p.read_text());r['status']=status;r['error']='synthetic typed failure';put(p,r)
    a=audit_run(run,None,root);assert a['status_counts'][status]==1
    assert a['portability_by_policy']['D0']['expected_jobs']==16 and a['portability_by_policy']['D0']['failure_fraction']==1/16
    assert a['by_policy']['D0']['eight']['K_complete']==0

def test_convergence_warnings_and_iteration_summaries_are_descriptive(tmp_path):
    root,run,jobs=synthetic_run(tmp_path);p=Path(jobs[0]['out']);r=json.loads(p.read_text());r.update(convergence_warning_count=2,warning_count=2,n_iter=[[8],[10]],warning_categories={'ConvergenceWarning':2});put(p,r)
    a=audit_run(run,None,root);c=next(x for x in a['convergence_summary'] if x['estimator']=='LR');assert c['convergence_warning_count']==2 and c['max_recorded_iterations']==10 and c['planned_jobs']==2

def test_full_two_edge_48_task_primary_replay_and_conservative_undefined_test(tmp_path):
    from synthetic_statistics_fixtures import add_synthetic_edge
    from scripts.aggregate_primary import aggregate_primary
    root,first,jobs=synthetic_run(tmp_path,n_tasks=48);second=add_synthetic_edge(root)
    a=tmp_path/'linear_audit.json';b=tmp_path/'ada_audit.json';audit_run(first,a,root);audit_run(second,b,root)
    result=aggregate_primary({'E_LINEAR_DUAL':a,'E_ADA_SAMME':b},root)
    assert result['family_size']==2 and result['policy']=='D0' and result['preprocessing']=='P1'
    assert result['status']=='PRIMARY_FAMILY_WITH_UNTESTABLE_EDGE'
    assert all(r['p_raw'] is None and r['p_holm']==1 and not r['reject_materiality_null'] for r in result['edges'].values())
    # Score summary forgery cannot pass simply by preserving lock metadata.
    forged=json.loads(a.read_text());forged['by_policy']['D0']['eight']['M_pp']=0.;put(a,forged)
    with pytest.raises(RuntimeError,match='supplied audit differs'):aggregate_primary({'E_LINEAR_DUAL':a,'E_ADA_SAMME':b},root)
