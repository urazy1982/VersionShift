"""No-fit extraction of manuscript result gaps from the validated PRIVATE ZIP.
Usage: python extract_missing_values.py ARCHIVE.zip NEW_OUTPUT_DIRECTORY
No network, model fitting, manuscript edits, or raw labels/predictions exported.
"""
from pathlib import Path
import sys,os,json,csv,zipfile,hashlib,collections,itertools,io,datetime,shutil
import numpy as np
ARCHIVE=Path(sys.argv[1]).resolve();OUT=Path(sys.argv[2]).resolve();OUT.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def put(n,x):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def table(n,rows):
 if not rows:return
 p=OUT/n
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert sha(ARCHIVE)=='f0e37943808537f757edde212ad53dafa833fe1d77393873aea1b7d59e3d9953'
Z=zipfile.ZipFile(ARCHIVE);manifest=json.loads(Z.read('SCIENTIFIC_FINAL_PACKAGE_SHA256.json'))
def read(n):
 raw=Z.read(n);assert hashlib.sha256(raw).hexdigest()==manifest[n];return json.loads(raw)
frozen=read('runs/plan/FROZEN_PROTOCOL.json');assert hashlib.sha256(Z.read('runs/plan/FROZEN_PROTOCOL.json')).hexdigest()=='7a62fb456f89dc58d2797e0e5ce83a456b178b4e740de93e6e9eacfeeca56f55'
# Use the byte-identical registered statistics implementation in an isolated import root.
for n in ['statistics.py','utils.py','__init__.py']:
 name='versionshift_v3/'+n;raw=Z.read(name);assert hashlib.sha256(raw).hexdigest()==manifest[name]
 if name in frozen['source_hashes']:assert hashlib.sha256(raw).hexdigest()==frozen['source_hashes'][name]
 p=OUT/'code'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
sys.path.insert(0,str(OUT/'code'))
from versionshift_v3.statistics import evaluate_edge,paired_mitigation,ALGOS
analysis=read('config/analysis.json');spec=read('config/statistical_spec_v305.json');allocation=read('runs/plan/PROSPECTIVE_TASK_ALLOCATION.json');E5=read('config/E5_EDGES.json')
progress=read('runs/execution_v305_20261117T164743Z/EXECUTION_PROGRESS.json');rawscores=read('reports/SCIENTIFIC_FINAL_VALIDATION_20261119/RAW_FOLD_SCORE_REPLAY.json')
score=collections.defaultdict(dict);foldvectors={};records={};states=collections.Counter();reasons=collections.Counter();raw_primary_replay=0;independent_groups=[]
for r in rawscores:
 key=(r['task_id'],r['release'],r['algo'],r['policy']);assert key not in score[r['group']];assert len(r['fold_balanced_accuracy_pp'])==30;score[r['group']][key]=float(np.mean(r['fold_balanced_accuracy_pp']));foldvectors[r['group'],r['job_id']]=r['fold_balanced_accuracy_pp']
def same(a,b):
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):return bool(np.isclose(a,b,rtol=1e-9,atol=1e-9,equal_nan=True))
 return a==b
def registered_metrics(sc,ids,rels,panel,policy):
 filtered={(t,v,a):x for (t,v,a,p),x in sc.items() if p==policy}
 out=evaluate_edge(filtered,algo_panel=panel,margin=analysis['primary_margin_pp'],bootstrap_reps=analysis['bootstrap_samples'],seed=analysis['bootstrap_seed'],expected_tasks=ids,releases=rels,spec=spec,winner_margins=analysis['winner_gap_margins_pp'])
 # Independent definitions for all reported conclusion outcomes; strict > matches frozen code.
 effects=[];winners=0;affected=0;anyaffected=0;rev=0;eligible=0;margins={float(m):[0,0] for m in analysis['winner_gap_margins_pp']};shifts=collections.defaultdict(list)
 for t in sorted(ids):
  if any((t,v,a) not in filtered for v in rels for a in panel):continue
  left={a:filtered[t,rels[0],a] for a in panel};right={a:filtered[t,rels[1],a] for a in panel};delta={a:right[a]-left[a] for a in panel};pairs=list(itertools.combinations(panel,2));dd=[delta[a]-delta[b] for a,b in pairs];m=float(np.mean(np.abs(dd)));effects.append(m);affected+=m>.5;anyaffected+=any(abs(x)>.5 for x in dd)
  ow=sorted(a for a in panel if abs(left[a]-max(left.values()))<=1e-12);nw=sorted(a for a in panel if abs(right[a]-max(right.values()))<=1e-12);changed=ow!=nw;winners+=changed
  for margin in margins:
   lo=sorted(left.values(),reverse=True);hi=sorted(right.values(),reverse=True)
   if len(ow)==len(nw)==1 and lo[0]-lo[1]>max(margin,1e-12) and hi[0]-hi[1]>max(margin,1e-12):margins[margin][0]+=1;margins[margin][1]+=changed
  for a,b in pairs:
   x=left[a]-left[b];y=right[a]-right[b]
   if abs(x)>.5 and abs(y)>.5:eligible+=1;rev+=x*y<0
  for a,x in delta.items():shifts[a].append(x)
 assert np.isclose(np.mean(effects),out['M_pp'],atol=1e-12) and winners==out['winner_changes'] and affected==out['affected'] and anyaffected==out['affected_for_equivalence']
 assert rev==out['material_reversal_summary']['reversals'] and eligible==out['material_reversal_summary']['material_gap_eligible_pairs']
 for m,(n,w) in margins.items():assert n==out['margin_qualified_winners'][str(m)]['eligible_denominator'] and w==out['margin_qualified_winners'][str(m)]['winner_changes']
 for a,values in shifts.items():assert np.isclose(np.mean(values),out['per_estimator_shifts'][a]['mean_shift_pp'],atol=1e-12)
 return out
summaries={};rows=[];marginrows=[];shiftrows=[];portability=[];contrasts=[];d2rows=[];p2rows=[];modern_taskrows=[];e5rows=[];fullgroups={};raw_hash_list=[]
for g in progress['runs']:
 name=g['name'];relbase='runs/results/'+name;lk=read(relbase+'/RUN_LOCK.json');assert set(lk['task_ids'])==set(g['task_ids']);rs={}
 for jid in lk['job_spec_digests']:
  path=relbase+'/results/'+jid+'.json';r=read(path);assert r['job_id']==jid and r['job_spec_sha256']==lk['job_spec_digests'][jid];assert r['environment_lock_sha256']==lk['environment_lock_hashes'][r['release']];rs[jid]=r;states[r['status']]+=1
  if r['status']=='SUPPORTED':
   assert np.allclose(foldvectors[name,jid],r['balanced_accuracy_pp'],rtol=0,atol=1e-10)
   predpath=relbase+'/results/'+jid+'.npz';assert manifest[predpath]==r['prediction_archive_sha256']
   # Fresh no-fit scoring of BOTH complete primary families from actual archive predictions.
   if g['edge'] in spec['primary_family'] and g['frame']=='evaluation' and g['preprocessing']=='P1' and r['policy']=='D0':
    raw=Z.read(predpath);assert hashlib.sha256(raw).hexdigest()==r['prediction_archive_sha256'];scores=[]
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:
     for f in r['fold_keys']:
      y=z[f+'_y'];p=z[f+'_pred'];scores.append(100.*float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)])))
    assert np.allclose(scores,foldvectors[name,jid],rtol=0,atol=1e-10);raw_primary_replay+=1
  else:reasons[(g['edge'],g['frame'],g['preprocessing'],r['release'],r['algo'],r['policy'],r['status'],r.get('error',''))]+=1
 records[name]=rs;rels=list(lk['release_pythons']);sc=score[name];ids=g['task_ids'];primary=g['edge'] in spec['primary_family'];panel=ALGOS if primary else ALGOS[:6]
 m=registered_metrics(sc,ids,rels,panel,'D0');summaries[name]={'D0_full_registered_panel':m};fullgroups[name]=g
 ap='runs/audit/'+Path(g['audit']).name;official=read(ap);label='eight' if primary else 'legacy_six';assert same(m,official['by_policy']['D0'][label]),'Frozen audit mismatch '+name
 basic={'group':name,'edge':g['edge'],'frame':g['frame'],'preprocessing':g['preprocessing'],'policy':'D0','selected_tasks':len(ids),'complete_tasks':m['K_complete'],'estimators':len(panel),'pairs':m['pair_count'],'M_pp':m['M_pp'],'bootstrap_low_pp':m['M_boot_ci95'][0],'bootstrap_high_pp':m['M_boot_ci95'][1],'affected_M_gt_0p5':m['affected'],'affected_any_pair_gt_0p5':m['affected_for_equivalence'],'winner_changes':m['winner_changes'],'winner_selected_denominator':len(ids),'material_reversals':m['material_reversal_summary']['reversals'],'material_eligible_pairs':m['material_reversal_summary']['material_gap_eligible_pairs'],'selected_task_pair_opportunities':m['material_reversal_summary']['selected_task_pair_opportunities'],'equivalence_status':m['equivalence']['status'],'secondary_inference':'DESCRIPTIVE_ONLY','release_old':m['old'],'release_new':m['new'],'environment_treatment':'MODERN_MATCHED_COMMON_STACK' if primary else 'HISTORICAL_FIXED_STACK' if 'FIXED_STACK' in name else 'HISTORICAL_RELEASE_ERA_STACKS'};rows.append(basic)
 for margin,v in m['margin_qualified_winners'].items():marginrows.append({'group':name,'edge':g['edge'],'frame':g['frame'],'preprocessing':g['preprocessing'],'policy':'D0','panel_size':len(panel),'margin_strictly_gt_pp':margin,'selected_tasks':len(ids),'eligible_tasks':v['eligible_denominator'],'winner_changes':v['winner_changes']})
 for a,v in m['per_estimator_shifts'].items():shiftrows.append({'group':name,'edge':g['edge'],'frame':g['frame'],'preprocessing':g['preprocessing'],'policy':'D0','algo':a,'selected_tasks':len(ids),'paired_tasks':len(v['task_ids']),'mean_new_minus_old_pp':v['mean_shift_pp'],'mean_incident_pair_contribution_to_M_pp':m['algorithm_leverage_mean_pp'][a]})
 if g['edge'].startswith('E5_'):e5rows.append(basic)
 if primary:
  c6=registered_metrics(sc,ids,rels,ALGOS[:6],'D0');summaries[name]['D0_six_estimator_continuity']=c6;assert same(c6,official['by_policy']['D0']['legacy_six'])
  shared=[a for a in ALGOS if all((t,v,a,'D1') in sc for t in ids for v in rels)];assert shared==[a for a in ALGOS if a!='HGB'];d0=registered_metrics(sc,ids,rels,shared,'D0');d1=registered_metrics(sc,ids,rels,shared,'D1');mit=paired_mitigation(d0,d1,seed=analysis['bootstrap_seed'],bootstrap_reps=analysis['bootstrap_samples']);assert mit['paired_tasks']==len(ids)
  summaries[name].update(D0_shared_seven=d0,D1_shared_seven=d1,D0_minus_D1_shared_seven_paired=mit)
  contrasts.append({'group':name,'edge':g['edge'],'frame':g['frame'],'preprocessing':g['preprocessing'],'selected_tasks':len(ids),'paired_tasks':mit['paired_tasks'],'shared_estimators':';'.join(shared),'shared_panel_size':len(shared),'shared_pairs':21,'D0_shared_M_pp':d0['M_pp'],'D1_shared_M_pp':d1['M_pp'],'D0_minus_D1_reduction_pp':mit['reduction_pp'],'reduction_bootstrap_low_pp':mit['reduction_ci95_pp'][0],'reduction_bootstrap_high_pp':mit['reduction_ci95_pp'][1],'D0_shared_winner_changes':d0['winner_changes'],'D1_shared_winner_changes':d1['winner_changes'],'D1_missing_original_estimator':'HGB','D0_full_eight_M_pp':m['M_pp'],'D0_legacy_six_M_pp':c6['M_pp'],'interpretation':'descriptive matched seven-estimator/21-pair contrast; HGB not imputed'})
  for policy in lk['policies']:
   for a in ALGOS:
    for v in rels:
     selected=[r for r in rs.values() if r['policy']==policy and r['algo']==a and r['release']==v];assert len(selected)==len(ids);cnt=collections.Counter(r['status'] for r in selected)
     portability.append({'group':name,'edge':g['edge'],'frame':g['frame'],'preprocessing':g['preprocessing'],'policy':policy,'algo':a,'release':v,'selected_tasks':len(ids),'supported':cnt['SUPPORTED'],'unsupported':cnt['UNSUPPORTED'],'removed_behavior':cnt['REMOVED_BEHAVIOR'],'other_status':sum(n for k,n in cnt.items() if k not in ['SUPPORTED','UNSUPPORTED','REMOVED_BEHAVIOR']),'reason':next((r.get('error','') for r in selected if r['status']!='SUPPORTED'),'')})
  mech='LinearSVC' if g['edge']=='E_LINEAR_DUAL' else 'AdaBoost'
  for policy in ['D0','D1','D2']:
   available=[t for t in ids if all((t,v,mech,policy) in sc for v in rels)]
   values=[sc[t,rels[1],mech,policy]-sc[t,rels[0],mech,policy] for t in available]
   d2rows.append({'group':name,'edge':g['edge'],'frame':g['frame'],'preprocessing':g['preprocessing'],'contrast':'new_minus_old_same_policy','algo':mech,'policy':policy,'release':rels[0]+' -> '+rels[1],'selected_tasks':len(ids),'paired_tasks':len(available),'mean_score_difference_pp':float(np.mean(values)) if values else None,'bootstrap_low_pp':float(np.percentile(np.asarray(values)[np.random.RandomState(20261116).randint(0,len(values),(20000,len(values)))].mean(axis=1),2.5)) if values else None,'bootstrap_high_pp':float(np.percentile(np.asarray(values)[np.random.RandomState(20261116).randint(0,len(values),(20000,len(values)))].mean(axis=1),97.5)) if values else None,'status':'SUPPORTED' if len(available)==len(ids) else 'REMOVED_BEHAVIOR_NO_CROSS_RELEASE_ESTIMATE','note':'Single-estimator score shift, not pairwise-panel M; no substitution for removed SAMME.R'})
  if mech=='AdaBoost':
   for v in rels:
    for left,right in [('D0','D1'),('D0','D2')]:
     avail=[t for t in ids if (t,v,mech,left) in sc and (t,v,mech,right) in sc];values=[sc[t,v,mech,right]-sc[t,v,mech,left] for t in avail]
     d2rows.append({'group':name,'edge':g['edge'],'frame':g['frame'],'preprocessing':g['preprocessing'],'contrast':right+'_minus_'+left+'_same_release','algo':mech,'policy':left+' -> '+right,'release':v,'selected_tasks':len(ids),'paired_tasks':len(avail),'mean_score_difference_pp':float(np.mean(values)) if values else None,'bootstrap_low_pp':float(np.percentile(np.asarray(values)[np.random.RandomState(20261116).randint(0,len(values),(20000,len(values)))].mean(axis=1),2.5)) if values else None,'bootstrap_high_pp':float(np.percentile(np.asarray(values)[np.random.RandomState(20261116).randint(0,len(values),(20000,len(values)))].mean(axis=1),97.5)) if values else None,'status':'SUPPORTED' if len(avail)==len(ids) else 'REMOVED_BEHAVIOR_NO_WITHIN_RELEASE_ESTIMATE','note':'Old-release D1 SAMME versus D0/D2 SAMME.R; new-release D2 absent. Empty means unavailable, never zero.'})
  for t in ids:
   mt=next(r for r in m['per_task'] if r['task_id']==t);win=next(r for r in m['winner_identities'] if r['task_id']==t);modern_taskrows.append({'group':name,'edge':g['edge'],'frame':g['frame'],'preprocessing':g['preprocessing'],'task_id':t,'policy':'D0','panel_size':8,'M_pp':mt['M_pp'],'affected_M_gt_0p5':mt['affected_M_gt_margin'],'affected_any_pair_gt_0p5':mt['affected_any_pair'],'old_winners':';'.join(win['old_winners']),'new_winners':';'.join(win['new_winners']),'winner_changed':win['winner_changed']})
 independent_groups.append({'group':name,'registered_D0_audit_reproduced':True,'independent_winner_margin_reversal_affected_shift_checks':True,'selected_tasks':len(ids),'complete_tasks':m['K_complete']});print('EXTRACTED',name,flush=True)
assert states=={'SUPPORTED':23994,'UNSUPPORTED':5312,'REMOVED_BEHAVIOR':166} and len(independent_groups)==33 and len(e5rows)==13 and raw_primary_replay==1536
# Matched P2-P1 task-M contrasts; identical eight-estimator D0 panels.
for edge in spec['primary_family']:
 for frame in ['pilot','evaluation','external']:
  left=summaries[f'{edge}_{frame}_P1_D0-D1-D2']['D0_full_registered_panel'];right=summaries[f'{edge}_{frame}_P2_D0-D1-D2']['D0_full_registered_panel'];l={r['task_id']:r['M_pp'] for r in left['per_task']};rr={r['task_id']:r['M_pp'] for r in right['per_task']};assert l.keys()==rr.keys();values=np.array([rr[t]-l[t] for t in sorted(l)]);boots=values[np.random.RandomState(20261116).randint(0,len(values),(20000,len(values)))].mean(axis=1);p2rows.append({'edge':edge,'frame':frame,'selected_tasks':len(l),'paired_tasks':len(l),'panel_size':8,'P1_M_pp':left['M_pp'],'P2_M_pp':right['M_pp'],'P2_minus_P1_pp':float(values.mean()),'bootstrap_low_pp':float(np.percentile(boots,2.5)),'bootstrap_high_pp':float(np.percentile(boots,97.5)),'P1_winner_changes':left['winner_changes'],'P2_winner_changes':right['winner_changes'],'secondary_inference':'DESCRIPTIVE_ONLY'})
put('ALL_REGISTERED_EXTRACTED_STATISTICS.json',summaries)
table('all_33_registered_D0_summaries.csv',rows)
table('modern_primary_conclusion_outcomes.csv',[r for r in rows if r['edge'] in spec['primary_family'] and r['frame']=='evaluation' and r['preprocessing']=='P1'])
table('modern_D0_margin_qualified_winners.csv',[r for r in marginrows if r['edge'] in spec['primary_family']])
table('modern_D0_per_estimator_shifts.csv',[r for r in shiftrows if r['edge'] in spec['primary_family']])
table('modern_D1_shared_support_contrasts.csv',contrasts);table('modern_policy_portability.csv',portability);table('D2_mechanism_score_contrasts.csv',d2rows);table('modern_P2_sensitivity.csv',p2rows)
table('external_20_task_findings.csv',[r for r in rows if r['frame']=='external']);table('E5_all_13_historical_edges.csv',e5rows);table('modern_D0_task_level_conclusions.csv',modern_taskrows)
table('unsupported_and_removed_reasons.csv',[{'edge':k[0],'frame':k[1],'preprocessing':k[2],'release':k[3],'algo':k[4],'policy':k[5],'status':k[6],'error':k[7],'records':v} for k,v in sorted(reasons.items())]);table('historical_margin_qualified_winners.csv',[r for r in marginrows if r['edge']=='E_ORIGINAL'])
# Exact 13 reviewed manuscript placeholders: 12 are findings; the final one requests a new DOI.
markers=[('TV01','D1-HGB support','PASS','modern_policy_portability.csv; unsupported_and_removed_reasons.csv'),('TV02','Affected counts','PASS','modern_primary_conclusion_outcomes.csv'),('TV03','Strict winner changes','PASS','modern_primary_conclusion_outcomes.csv'),('TV04','Margin-qualified winners','PASS','modern_D0_margin_qualified_winners.csv'),('TV05','Material pair reversals','PASS','modern_primary_conclusion_outcomes.csv'),('TV06','Per-estimator shifts','PASS','modern_D0_per_estimator_shifts.csv'),('TV07','Six-estimator continuity M','PASS','modern_D1_shared_support_contrasts.csv: D0_legacy_six_M_pp'),('TV08','D1 matched-panel contrasts','PASS','modern_D1_shared_support_contrasts.csv'),('TV09','D2 mechanism contrasts','PASS','D2_mechanism_score_contrasts.csv'),('TV10','External 20-task findings','PASS','external_20_task_findings.csv'),('TV11','All 13 E5 edge effects','PASS','E5_all_13_historical_edges.csv'),('TV12','Mechanism/concentration interpretation','PASS','modern_D0_per_estimator_shifts.csv; D2_mechanism_score_contrasts.csv; contrast notes'),('TV13','New v3.0.5 Zenodo DOI / public predictions','NOT_AVAILABLE_NOT_A_RESULT','Old DOI 10.5281/zenodo.24080651 is v2.6 only. No new DOI created and private prediction archive not published.')]
table('TO_VERIFY_RESOLUTION.csv',[dict(zip(['marker','item','status','evidence'],r)) for r in markers])
shutil.copy2(Path(__file__),OUT/'extract_missing_values.py')
put('EXTRACTION_VERIFICATION.json',{'status':'PASS_ALL_AVAILABLE_NUMERIC_ITEMS','archive_sha256':sha(ARCHIVE),'frozen_sha256':'7a62fb456f89dc58d2797e0e5ce83a456b178b4e740de93e6e9eacfeeca56f55','all_archive_member_hashes_preverified':112841,'groups':independent_groups,'status_counts':dict(states),'primary_prediction_archives_freshly_rescored':raw_primary_replay,'primary_raw_fold_scores_freshly_rescored':raw_primary_replay*30,'all_other_score_provenance':'Previously independently raw-replayed fold score vectors in hash-verified final archive; additionally matched each result JSON and registered audit. New independent implementations checked all reported winner/margin/reversal/affected/shift outcomes.','model_fits':0,'new_predictions':0,'manuscript_modified':False,'public_uploads':False,'threshold_rules':{'winner_tie_tolerance_pp':1e-12,'margin_winner_eligibility':'unique winners and BOTH endpoint leading gaps strictly > margin','material_reversal':'both endpoint gaps strictly > 0.5 pp in magnitude and sign reverses','affected_descriptor':'M_t > 0.5 pp','affected_equivalence':'any absolute signed pair delta > 0.5 pp'},'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print('EXTRACTION_COMPLETE',str(OUT),flush=True)
