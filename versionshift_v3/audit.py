"""Independent source-row prediction replay and frozen task-level inference."""
import argparse,collections
from pathlib import Path
import numpy as np
from .utils import read_json,sha256,write_once,require,digest_json
from .statistics import ALGOS,evaluate_edge,paired_mitigation
from .binding import load_settings,source_binding,frozen_source_rows,identity_fields

STATES={'SUPPORTED','UNSUPPORTED','REMOVED_BEHAVIOR','INEFFECTIVE','API_REMOVED','FIT_FAILED','TIMED_OUT','MISSING_JOB'}
NONPORTABLE={'UNSUPPORTED','REMOVED_BEHAVIOR','INEFFECTIVE','API_REMOVED'}

def score_replay(rec, expected_job=None, return_details=False):
    """Recompute recalls without sklearn's metric; bind every prediction to raw rows."""
    r=read_json(rec)
    if r['status']!='SUPPORTED':return None
    binding=r.get('source_binding');require(isinstance(binding,dict),'source binding absent from prediction record')
    if expected_job is not None:require(binding==expected_job['source_binding'],'prediction source differs from expected job')
    for key in ('matrix_manifest','labels','row_positions','provenance'):
        require(sha256(binding[key+'_path'])==binding[key+'_sha256'],'source replay SHA mismatch')
    mf=Path(binding['matrix_manifest_path']);manifest=read_json(mf)
    require(manifest['task_id']==r['task_id'],'source replay task identity differs')
    source_y=np.load(binding['labels_path'],allow_pickle=False);positions=np.load(binding['row_positions_path'],allow_pickle=False)
    require(source_y.ndim==positions.ndim==1 and len(source_y)==len(positions) and positions.dtype.kind in 'iu' and len(np.unique(positions))==len(positions),'invalid source rows')
    z=Path(rec).with_suffix('.npz');require(z.exists() and sha256(z)==r.get('prediction_archive_sha256'),'prediction SHA mismatch '+str(rec))
    expected={f'r{rep}_f{fold}' for rep in range(6) for fold in range(5)};keys=r.get('fold_keys',[])
    require(len(keys)==30 and set(keys)==expected,'not exact 30-fold universe '+str(rec))
    entries={e['key']:e for e in manifest['folds'] if e['preprocessing_id']==r['preprocessing_id']}
    require(set(entries)==expected and len([e for e in manifest['folds'] if e['preprocessing_id']==r['preprocessing_id']])==30,'duplicate/missing source folds')
    scores=[];folds={};repetitions=collections.defaultdict(list)
    with np.load(z,allow_pickle=False) as archive:
        require(set(archive.files)=={k+s for k in keys for s in ('_y','_pred','_test_indices','_source_row_ids')},'prediction archive schema differs')
        for key in keys:
            e=entries[key];matrix=(mf.parent/e['file']).resolve();require(matrix.parent==mf.parent.resolve(),'matrix path escapes task directory')
            require(sha256(matrix)==e['sha256'],'source matrix SHA mismatch')
            with np.load(matrix,allow_pickle=False) as m:
                ti=m['test_indices'];tr=m['train_indices'];yt=archive[key+'_y'];yp=archive[key+'_pred'];idx=archive[key+'_test_indices'];ids=archive[key+'_source_row_ids']
                require(ti.dtype.kind in 'iu' and tr.dtype.kind in 'iu' and np.array_equal(np.sort(np.concatenate([tr,ti])),np.arange(len(source_y))),'invalid frozen row partition')
                require(np.array_equal(idx,ti) and np.array_equal(ids,positions[ti]),'prediction row identity mismatch')
                require(np.array_equal(yt,source_y[ti]) and np.array_equal(m['y_test'],source_y[ti]) and np.array_equal(m['y_train'],source_y[tr]),'prediction/source label mismatch')
                require(yt.ndim==yp.ndim==1 and len(yt)==len(yp) and len(yt)>0 and np.isin(yp,np.unique(source_y)).all(),'invalid prediction labels/length')
                scores.append(float(100*np.mean([np.mean(yp[yt==c]==c) for c in np.unique(yt)])))
                folds[key]={'pred':yp.copy(),'y':yt.copy(),'indices':idx.copy(),'source_ids':ids.copy()};repetitions[key.split('_')[0]].append(ti.copy())
    for rep,parts in repetitions.items():require(np.array_equal(np.sort(np.concatenate(parts)),np.arange(len(source_y))),'repetition does not partition source rows')
    require(len(r['balanced_accuracy_pp'])==30 and np.allclose(scores,r['balanced_accuracy_pp'],atol=1e-10,rtol=0),'score replay mismatch '+str(rec))
    result={'score_pp':float(np.mean(scores)),'folds':folds}
    return result if return_details else result['score_pp']

def verify_run_identity(root,run_dir,lock):
    from .run import jobs_for_edge,verify_frozen_sources
    root=Path(root).resolve();run_dir=Path(run_dir).resolve();freeze_path=root/'runs/plan/FROZEN_PROTOCOL.json';freeze=read_json(freeze_path)
    require(sha256(freeze_path)==lock['frozen_sha256'],'run freeze digest differs');verify_frozen_sources(root,freeze)
    analysis,spec,hashes=load_settings(root,freeze['source_hashes']);require(lock['analysis_hashes']==hashes,'run analysis configuration differs')
    plan_path=root/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json';plan=read_json(plan_path)
    require(sha256(plan_path)==freeze['plan_sha256']==lock['selected_allocation_sha256'],'run allocation differs')
    frame_key={'pilot':'pilot_task_ids','evaluation':'evaluation_task_ids','external':'external_task_ids'}
    require(lock['frame'] in frame_key and lock['task_ids']==plan[frame_key[lock['frame']]],'run selected task identities differ')
    require(len(lock['policies'])==len(set(lock['policies'])) and set(lock['policies'])<= {'D0','D1','D2'},'unregistered or duplicate policies')
    jobs=list(jobs_for_edge(root,plan,lock['edge'],lock['frame'],lock['preprocessing'],lock['policies'],lock['release_pythons'],run_dir,lock['frozen_sha256']))
    for j in jobs:
        j['environment_lock_sha256']=lock['environment_lock_hashes'][j['release']];frozen_source_rows(root,j['source_binding'],freeze)
    require(set(lock['release_pythons'])=={j['release'] for j in jobs},'run releases differ from edge')
    require(len(jobs)==lock['jobs'] and digest_json([j['job_id'] for j in jobs])==lock['job_ids_sha256'],'expected job universe differs')
    require({j['job_id']:digest_json(j) for j in jobs}==lock['job_spec_digests'],'job spec lock differs')
    require(lock['worker_sha256']==sha256(root/'versionshift_v3/worker.py'),'run worker identity differs')
    return jobs,analysis,spec,hashes

def prediction_disagreement(left,right):
    require(set(left['folds'])==set(right['folds']),'prediction mismatch fold universe differs')
    diffs=[];total=0;bad=0
    for key in sorted(left['folds']):
        a=left['folds'][key];b=right['folds'][key]
        require(np.array_equal(a['source_ids'],b['source_ids']) and np.array_equal(a['y'],b['y']),'cross-release prediction rows differ')
        k=int(np.count_nonzero(a['pred']!=b['pred']));n=len(a['pred']);diffs.append(k/n);bad+=k;total+=n
    return {'mean_fold_disagreement_rate':float(np.mean(diffs)),'prediction_disagreements':bad,'prediction_occurrences':total,
            'pooled_disagreement_rate':bad/total,'unit':'matched prediction occurrences over 6x5 repeated folds, not independent observations'}

def audit_run(run_dir,out,project=None):
    root=Path(project or Path(__file__).resolve().parents[1]).resolve();run_dir=Path(run_dir).resolve();lock=read_json(run_dir/'RUN_LOCK.json')
    jobs,analysis,spec,hashes=verify_run_identity(root,run_dir,lock);expected={j['job_id']:j for j in jobs};seen={};records={};replays={};diagnostics=[]
    for p in sorted((run_dir/'results').glob('*.json')):
        r=read_json(p);jid=r.get('job_id');require(jid in expected and jid not in seen,'unexpected or duplicate job identity')
        j=expected[jid];require(p.resolve()==Path(j['out']).resolve(),'result filename differs from expected job')
        require(all(r.get(k)==v for k,v in identity_fields(j).items()),'result identity differs from run lock')
        require(r.get('job_spec_sha256')==digest_json(j),'result spec digest differs');require(r.get('worker_sha256')==lock['worker_sha256'],'result worker differs')
        require(r['status'] in STATES,'unknown portability/failure state');seen[jid]=r
        if r['status']=='SUPPORTED':
            replay=score_replay(p,j,True);key=(r['task_id'],r['release'],r['algo'],r['policy']);records[key]=replay['score_pp'];replays[key]=replay
        diagnostics.append({'job_id':jid,'task_id':r['task_id'],'release':r['release'],'estimator':r['algo'],'policy':r['policy'],'status':r['status'],
                            'error':r.get('error'),'warning_count':r.get('warning_count',0),'warnings':r.get('warnings',[]),'iterations':r.get('n_iter',[]),
                            'convergence_warning_count':r.get('convergence_warning_count',0),'warning_categories':r.get('warning_categories',{}),
                            'linear_svc_observations':r.get('linear_svc_effective_solver_by_fold',[])})
    for jid,j in expected.items():
        if jid not in seen:
            seen[jid]={**identity_fields(j),'status':'MISSING_JOB'};diagnostics.append({'job_id':jid,'status':'MISSING_JOB','error':'expected result absent'})
    planned_algos=[a for a in ALGOS if a in {j['algo'] for j in jobs}]
    releases=sorted({j['release'] for j in jobs});counts=collections.Counter(r['status'] for r in seen.values());by_policy={};portability={}
    for policy in lock['policies']:
        per=[r for r in seen.values() if r['policy']==policy];cs=collections.Counter(r['status'] for r in per)
        portability[policy]={'expected_jobs':len(per),'statuses':{s:cs[s] for s in sorted(STATES)},'supported_jobs':cs['SUPPORTED'],
                             'nonportable_jobs':sum(cs[s] for s in NONPORTABLE),'supported_fraction':cs['SUPPORTED']/len(per),
                             'failure_fraction':1-cs['SUPPORTED']/len(per),'denominator':'all planned release x task x estimator jobs for policy'}
        src={(t,v,a):s for (t,v,a,p),s in records.items() if p==policy};by_policy[policy]={}
        for label,panel in [('eight',ALGOS),('legacy_six',ALGOS[:6])]:
            if not set(panel)<=set(planned_algos):
                by_policy[policy][label]={'status':'NOT_PLANNED_FOR_EDGE','panel':panel,'per_task':[],'selected_task_ids':lock['task_ids'],'scope':'not counted as a portability failure'};continue
            kwargs=dict(margin=analysis['primary_margin_pp'],bootstrap_reps=analysis['bootstrap_samples'],seed=analysis['bootstrap_seed'],
                        expected_tasks=lock['task_ids'],releases=releases,spec=spec,winner_margins=analysis['winner_gap_margins_pp'])
            if not any(all((t,v,a) in src for v in releases for a in panel) for t in lock['task_ids']):
                by_policy[policy][label]={'status':'NOT_ESTIMABLE','K_complete':0,'K_incomplete':len(lock['task_ids']),
                                        'selected_task_ids':lock['task_ids'],'selected_denominator':len(lock['task_ids']),
                                        'incomplete_tasks':lock['task_ids'],'task_failure_fraction':1.,'per_task':[],
                                        'panel':panel,'scope':'all required paired cells absent/nonportable; no zero imputation'}
            else:by_policy[policy][label]=evaluate_edge(src,panel,**kwargs)
    disagreements=[]
    for policy in lock['policies']:
        for task in lock['task_ids']:
            for algo in planned_algos:
                a=replays.get((task,releases[0],algo,policy));b=replays.get((task,releases[1],algo,policy))
                if a is not None and b is not None:disagreements.append({'task_id':task,'policy':policy,'estimator':algo,**prediction_disagreement(a,b)})
    convergence=[]
    for policy in lock['policies']:
        for algo in sorted({j['algo'] for j in jobs}):
            rr=[r for r in seen.values() if r['policy']==policy and r['algo']==algo]
            iterations=[float(v) for r in rr for fold in r.get('n_iter',[]) for v in np.asarray(fold).ravel() if np.isfinite(v)]
            convergence.append({'policy':policy,'estimator':algo,'planned_jobs':len(rr),
                                'statuses':dict(collections.Counter(r['status'] for r in rr)),
                                'warning_count':sum(r.get('warning_count',0) for r in rr),
                                'convergence_warning_count':sum(r.get('convergence_warning_count',0) for r in rr),
                                'jobs_with_convergence_warning':sum(r.get('convergence_warning_count',0)>0 for r in rr),
                                'recorded_iteration_values':len(iterations),'max_recorded_iterations':max(iterations) if iterations else None,
                                'interpretation':'warning diagnostics, not proof that all other jobs converged'})
    mitigation={}
    if 'D0' in by_policy and 'D1' in by_policy:
        for panel in ('eight','legacy_six'):mitigation[panel]=paired_mitigation(by_policy['D0'][panel],by_policy['D1'][panel],analysis['bootstrap_seed'],analysis['bootstrap_samples'])
    report={'status':'RAW_SOURCE_ROW_REPLAY_PASS' if not counts['MISSING_JOB'] else 'RAW_SOURCE_ROW_REPLAY_PASS_WITH_MISSING_JOBS',
            'run_dir':str(run_dir),'run_lock_sha256':sha256(run_dir/'RUN_LOCK.json'),'identity':lock,
            'analysis_hashes':hashes,'analysis_settings':analysis,'statistical_spec':spec,
            'expected_job_records':len(jobs),'observed_job_records':len(list((run_dir/'results').glob('*.json'))),
            'status_counts':{s:counts[s] for s in sorted(STATES)},'portability_by_policy':portability,
            'by_policy':by_policy,'mitigation':mitigation,'prediction_disagreement':disagreements,
            'prediction_pair_denominator':{'eligible_pairs':len(disagreements),'selected_task_ids':lock['task_ids'],'per_estimator_policy':[{'policy':p,'estimator':a,'paired_tasks':sum(x['policy']==p and x['estimator']==a for x in disagreements),'selected_tasks':len(lock['task_ids'])} for p in lock['policies'] for a in planned_algos]},
            'convergence_summary':convergence,'job_diagnostics':diagnostics,'score_replayed':len(records)*30,'scientific_claim_status':'CONDITIONAL_FRAME_ONLY'}
    if out is not None:write_once(out,report)
    return report

def main():
    q=argparse.ArgumentParser();q.add_argument('--run',required=True);q.add_argument('--out',required=True);q.add_argument('--project')
    a=q.parse_args();r=audit_run(a.run,a.out,a.project);print(r['status'],r['observed_job_records'],'observed jobs')
if __name__=='__main__':main()
