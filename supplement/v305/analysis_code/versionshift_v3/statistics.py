"""Task-level protocol statistics. No model fitting or outcome IO."""
import itertools
import numpy as np
from scipy import stats
from .utils import require

ALGOS=['LR','SVC','kNN','RF','GB','HGB','LinearSVC','AdaBoost']

def t_inference(values, margin=0.5, alpha=0.05, confidence=0.90):
    a=np.asarray(values,dtype=float);require(a.ndim==1 and np.isfinite(a).all(),'nonfinite inference input')
    n=len(a);mean=float(a.mean()) if n else None
    result={'n':n,'mean_pp':mean,'status':'INSUFFICIENT_TASKS','p_one_sided':None,
            'one_sided_lower_pp':None,'ci_two_sided_pp':None,'df':n-1 if n else None}
    if n<2:return result
    sd=float(a.std(ddof=1));result['sample_sd_pp']=sd
    if np.all(a==a[0]) or sd==0.0:
        result['status']='ZERO_VARIANCE_T_UNDEFINED';return result
    se=sd/np.sqrt(n);t=float((mean-margin)/se)
    result.update(status='T_INFERENCE_DEFINED',t=t,p_one_sided=float(stats.t.sf(t,n-1)),
                  one_sided_lower_pp=float(mean-stats.t.ppf(1-alpha,n-1)*se),
                  ci_two_sided_pp=[float(mean-stats.t.ppf((1+confidence)/2,n-1)*se),
                                   float(mean+stats.t.ppf((1+confidence)/2,n-1)*se)])
    return result

def prevalence_bound(k,n,confidence=.95):
    require(isinstance(k,int) and isinstance(n,int) and 0<=k<=n,'invalid prevalence counts')
    if not n:return {'status':'NO_TASKS','upper':None,'affected':k,'denominator':n}
    upper=1.0 if k==n else float(stats.beta.ppf(confidence,k+1,n-k))
    return {'status':'CONDITIONAL_EXACT_BINOMIAL','affected':k,'denominator':n,'upper':upper,
            'confidence':confidence,'method':'one-sided Clopper-Pearson',
            'assumption':'iid Bernoulli task sampling; not established merely by metadata allocation'}

def holm_two(pvalues):
    require(len(pvalues)==2,'primary family must contain exactly two edges')
    require(all(np.isfinite(p) and 0<=p<=1 for p in pvalues.values()),'invalid primary p value')
    order=sorted(pvalues,key=lambda k:(pvalues[k],k));out={};last=0.
    for i,key in enumerate(order):
        last=max(last,min(1.,(2-i)*pvalues[key]));out[key]=last
    return out

def evaluate_edge(records,algo_panel=ALGOS,margin=.5,bootstrap_reps=20000,seed=20261116,
                  expected_tasks=None,releases=None,spec=None,winner_margins=(0.,.25,.5,1.)):
    spec=spec or {};alpha=spec.get('primary_alpha',.05);confidence=spec.get('pair_ci_confidence',.9)
    tie=spec.get('winner_tie_tolerance_pp',1e-12);reversal_margin=spec.get('material_reversal_margin_pp',margin)
    rule=spec.get('affected_rule','any_absolute_signed_pair_delta_gt_margin')
    require(rule in ('any_absolute_signed_pair_delta_gt_margin','M_gt_margin'),'unknown affected rule')
    require(len(algo_panel)>=2 and len(set(algo_panel))==len(algo_panel),'invalid estimator panel')
    releases=list(releases or sorted({k[1] for k in records}));require(len(releases)==2 and releases[0]!=releases[1],'expected exactly two releases')
    old,new=releases;tasks=sorted(expected_tasks if expected_tasks is not None else {k[0] for k in records})
    require(len(tasks)==len(set(tasks)),'duplicate selected tasks')
    require(all(k[0] in tasks and k[1] in releases for k in records),'record outside selected frame/releases')
    pairs=list(itertools.combinations(algo_panel,2));deltas=[];effects=[];winners=[];incomplete=[];shifts={a:[] for a in algo_panel}
    margin_winners={str(float(m)):{'margin_pp':float(m),'eligible_tasks':[],'changed_tasks':[]} for m in winner_margins}
    reversals=[];leverage=[];material_pair_eligible=0
    for task in tasks:
        left={a:records.get((task,old,a)) for a in algo_panel};right={a:records.get((task,new,a)) for a in algo_panel}
        if any(v is None for v in [*left.values(),*right.values()]):incomplete.append(task);continue
        require(all(np.isfinite(v) and 0<=v<=100 for v in [*left.values(),*right.values()]),'invalid percentage-point scores')
        dx={a:right[a]-left[a] for a in algo_panel};dd=np.array([dx[a]-dx[b] for a,b in pairs]);deltas.append(dd)
        for a in algo_panel:shifts[a].append(dx[a])
        M=float(np.mean(np.abs(dd)));any_pair=bool(np.any(np.abs(dd)>margin));affected=any_pair if rule=='any_absolute_signed_pair_delta_gt_margin' else M>margin
        effects.append({'task_id':task,'M_pp':M,'affected_M_gt_margin':M>margin,'affected_any_pair':any_pair,'affected_for_equivalence':affected,'affected':M>margin})
        lo=max(left.values());hi=max(right.values());ow=sorted(a for a in algo_panel if abs(left[a]-lo)<=tie);nw=sorted(a for a in algo_panel if abs(right[a]-hi)<=tie)
        winners.append({'task_id':task,'old_winners':ow,'new_winners':nw,'winner_changed':ow!=nw})
        old_scores=sorted(left.values(),reverse=True);new_scores=sorted(right.values(),reverse=True)
        for m in winner_margins:
            if len(ow)==len(nw)==1 and old_scores[0]-old_scores[1]>max(m,tie) and new_scores[0]-new_scores[1]>max(m,tie):
                row=margin_winners[str(float(m))];row['eligible_tasks'].append(task)
                if ow!=nw:row['changed_tasks'].append(task)
        task_reversals=[]
        for a,b in pairs:
            g0=left[a]-left[b];g1=right[a]-right[b]
            if min(abs(g0),abs(g1))>reversal_margin:material_pair_eligible+=1
            if g0*g1<0 and min(abs(g0),abs(g1))>reversal_margin:task_reversals.append({'pair':[a,b],'old_gap_pp':g0,'new_gap_pp':g1})
        reversals.append({'task_id':task,'count':len(task_reversals),'pairs':task_reversals})
        # Sum of incident-pair contributions equals M; split each pair equally.
        leverage.append({'task_id':task,'contribution_to_M_pp':{a:float(sum(abs(dx[a]-dx[b]) for b in algo_panel if b!=a)/(2*len(pairs))) for a in algo_panel},
                         'leave_one_out_M_reduction_pp':{a:float(M-np.mean([abs(dx[b]-dx[c]) for b,c in pairs if a not in (b,c)])) for a in algo_panel} if len(algo_panel)>2 else {a:None for a in algo_panel}})
    require(effects,'no complete-case task available')
    a=np.array([r['M_pp'] for r in effects]);dd=np.stack(deltas);n=len(a);N=len(tasks)
    require(isinstance(bootstrap_reps,int) and bootstrap_reps>0,'invalid bootstrap repetitions')
    draws=a[np.random.RandomState(seed).randint(0,n,size=(bootstrap_reps,n))].mean(axis=1)
    primary=t_inference(a,margin,alpha,confidence);signed=[]
    for i,pair in enumerate(pairs):
        inf=t_inference(dd[:,i],0.,alpha,confidence);ci=inf['ci_two_sided_pp']
        signed.append({'pair':list(pair),'mean_delta_pp':float(dd[:,i].mean()),'ci90_pp':ci,'confidence':confidence,
                       'inference_status':inf['status'],'within_practical_margin':bool(ci is not None and ci[0]>-margin and ci[1]<margin)})
    k=sum(r['affected_for_equivalence'] for r in effects);missing=N-n
    bound=prevalence_bound(int(k+missing),N,spec.get('prevalence_confidence',.95))
    eq_pairs=all(r['within_practical_margin'] for r in signed);eq_share=bound['upper']<spec.get('equivalence_max_affected_share',.1)
    justified=spec.get('prevalence_model_justified',False)
    eq='EQUIVALENT_CONDITIONAL_ON_TASK_MODEL' if eq_pairs and eq_share and justified and missing==0 else 'INCONCLUSIVE'
    for row in margin_winners.values():
        row['eligible_denominator']=len(row['eligible_tasks']);row['winner_changes']=len(row['changed_tasks']);row['selected_denominator']=N
    return {'evidence_scope':'FINITE_ALLOCATED_TASK_FRAME','population_inference_scope':'IID_TASK_MODEL_SENSITIVITY_ONLY',
            'inference_interpretation':'t, bootstrap and binomial bounds are conditional sensitivity analyses; the allocated48-task frame is not an iid population sample.',
            'old':old,'new':new,'panel':list(algo_panel),'pair_count':len(pairs),'selected_task_ids':tasks,
            'complete_task_ids':[r['task_id'] for r in effects],'selected_denominator':N,'K_complete':n,'K_incomplete':missing,
            'incomplete_tasks':incomplete,'task_failure_fraction':missing/N if N else None,'M_pp':float(a.mean()),
            'M_boot_ci95':np.percentile(draws,[2.5,97.5]).tolist(),'bootstrap_reps':bootstrap_reps,'bootstrap_seed':seed,
            'primary_test':primary,'p_one_sided_M_gt_margin':primary['p_one_sided'],
            'one_sided_lower95_pp':primary['one_sided_lower_pp'],'primary_alpha':alpha,
            'margin_pp':margin,'affected':sum(r['affected_M_gt_margin'] for r in effects),'affected_definition':'M_t > margin; legacy descriptor only',
            'affected_for_equivalence':k,'affected_rule':rule,'affected_share_finite_frame_interval':[k/N,(k+missing)/N],
            'affected_share_upper_bound':bound,'prevalence_model_justified':justified,
            'equivalence':{'status':eq,'all_signed_pairs_within_margin':eq_pairs,'share_bound_below_threshold':eq_share,
                           'assumptions_author_justified':justified,'missing_tasks':missing},
            'signed_pair_intervals':signed,'winner_changes':sum(r['winner_changed'] for r in winners),
            'winner_identities':winners,'winner_tie_tolerance_pp':tie,'margin_qualified_winners':margin_winners,
            'material_reversals':reversals,'material_reversal_margin_pp':reversal_margin,
            'material_reversal_summary':{'reversals':sum(r['count'] for r in reversals),'material_gap_eligible_pairs':material_pair_eligible,
                                         'complete_task_pair_opportunities':n*len(pairs),'selected_task_pair_opportunities':N*len(pairs),
                                         'reversal_rate_among_eligible':sum(r['count'] for r in reversals)/material_pair_eligible if material_pair_eligible else None},
            'per_estimator_shifts':{alg:{'mean_shift_pp':float(np.mean(v)),'task_shifts_pp':v,'task_ids':[r['task_id'] for r in effects],
                                       'inference':t_inference(v,0.,alpha,confidence)} for alg,v in shifts.items()},
            'algorithm_leverage':leverage,'algorithm_leverage_mean_pp':{alg:float(np.mean([row['contribution_to_M_pp'][alg] for row in leverage])) for alg in algo_panel},'per_task':effects,'secondary_status':'DESCRIPTIVE_NO_ADDITIONAL_CONFIRMATORY_FAMILY',
            'caution':'Finite allocated frame with prior historical exposure. Task-model inference is conditional; no all-upgrade prevalence claim.'}

def paired_mitigation(d0,d1,seed=20261116,bootstrap_reps=20000):
    a={r['task_id']:r['M_pp'] for r in d0['per_task']};b={r['task_id']:r['M_pp'] for r in d1['per_task']}
    selected=set(d0.get('selected_task_ids',a))|set(d1.get('selected_task_ids',b));common=sorted(set(a)&set(b));N=len(selected)
    base={'selected_denominator':N,'paired_tasks':len(common),'paired_task_ids':common,'excluded_task_ids':sorted(selected-set(common)),
          'D0_complete_denominator':len(a),'D1_complete_denominator':len(b),'paired_failure_fraction':1-len(common)/N if N else None,
          'pairing':'same task and same panel; unsupported cells never imputed'}
    if not common:return dict(base,status='NOT_IDENTIFIABLE')
    x=np.array([a[t]-b[t] for t in common]);old=np.array([a[t] for t in common]);idx=np.random.RandomState(seed).randint(0,len(x),(bootstrap_reps,len(x)))
    delta=x[idx].mean(axis=1);den=old[idx].mean(axis=1);good=den>0;atten=delta[good]/den[good]
    return dict(base,status='DESCRIPTIVE_PAIRED',reduction_pp=float(x.mean()),reduction_ci95_pp=np.percentile(delta,[2.5,97.5]).tolist(),
                relative_attenuation=float(x.mean()/old.mean()) if old.mean()>0 else None,
                attenuation_ci95=np.percentile(atten,[2.5,97.5]).tolist() if len(atten) else None,
                bootstrap_zero_denominator_count=int((~good).sum()))
