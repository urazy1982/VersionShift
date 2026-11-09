#!/usr/bin/env python3
"""VersionShift v2.2.5 CPU completion orchestration (external, additive).

Preserves immutable original controller/worker, frozen plan, Stage B selection/oracle,
and Stage B results. Runs remaining 47 datasets, deterministic A/A checks on the
16 pilot datasets, combines results into the full 63-dataset *finite-frame* analysis,
and analyses the 47 previously unused datasets separately.

NOTES:
  - full 63 incorporates 16 previously inspected exploratory Stage B datasets;
  - the 47 never-seen datasets are one below the pre-registered >=48 floor;
  - neither analysis may be advertised as untouched 63-dataset confirmation;
  - a local oracle freeze does not prove external timestamp / author approval.
"""
import argparse, hashlib, itertools, json, os, shutil, subprocess, sys, time
from pathlib import Path

ALGOS=('LR','SVC','kNN','RF','GB','HGB')
EDGES=(('POS','0.21.3','0.22.2.post1','EDGE_0213_0222'),
       ('NEG','1.8.0','1.9.1','EDGE_180_190'))
AA_VERSIONS=('0.21.3','0.22.2.post1','1.8.0','1.9.1')
EXPECTED_PLAN='523c42a3dabee44e66d43d538b228939b01fd7fe1b68285f542f0980906492f4'
EXPECTED_B='55554df012a30d7000d3b5840babd23f8e10b9daca57b10e8cf6099fce92943c'
EXPECTED_ORACLE='33f04fb6c6da89654f3524cfaef290644b80e37afcd5148dcffa740e9bcfe2f6'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def load(p):return json.loads(Path(p).read_text())
def fail(x):raise SystemExit('STOP: '+x)
def write_once(path,document):
    path=Path(path)
    payload=json.dumps(document,indent=2,sort_keys=True)+'\n'
    if path.exists():
        if path.read_text()!=payload: fail('immutable output differs: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(payload)

def context(args,require_selected=True):
    pr=Path(args.project).resolve(); dat=Path(args.data).resolve()
    plan=dat/'STAGE_B_PLAN_CACHED'; selection=dat/'STAGE_B_SELECTION.json'
    runs=dat/'STAGE_B_RUNS'; oracle=pr/'ORACLE_FROZEN.json'
    for p in (pr/'crossversion/controller.py',pr/'crossversion/worker.py',plan/'PLAN.json',selection,oracle,pr/'ORACLE_FRAME_REVIEWED.csv'):
        if not p.is_file():fail('required input missing: '+str(p))
    if sha(plan/'PLAN.json')!=EXPECTED_PLAN:fail('PLAN.json SHA-256 differs from reported frozen plan')
    if sha(selection)!=EXPECTED_B:fail('Stage B selection SHA-256 differs from frozen selection')
    if sha(pr/'ORACLE_FRAME_REVIEWED.csv')!=EXPECTED_ORACLE:fail('oracle reviewed CSV differs from validated reference')
    fz=load(oracle)
    if fz.get('status')!='FROZEN_COMPLETE' or fz.get('frame_sha256')!=EXPECTED_ORACLE:fail('oracle freeze invalid')
    frame=Path(fz['frame_path'])
    if not frame.is_absolute():frame=pr/frame
    if not frame.is_file() or sha(frame)!=EXPECTED_ORACLE:fail('frozen frame path invalid')
    pdoc=load(plan/'PLAN.json'); sdoc=load(selection)
    all_ids={d['name'] for d in pdoc['datasets']}
    b_ids=set(sdoc['stage_b_datasets'])
    if len(all_ids)!=63 or len(b_ids)!=16 or not b_ids<=all_ids:fail('dataset frame not 63 eligible / 16 pilot')
    if sdoc['plan_sha256']!=EXPECTED_PLAN:fail('Stage B selection not bound to frozen plan')
    remain=sorted(all_ids-b_ids)
    if len(remain)!=47:fail('expected 47 pilot-excluded datasets')
    csel=dat/'STAGE_C_HOLDOUT47_SELECTION.json'; output=dat/'STAGE_C_CPU_COMPLETION'
    if require_selected and not csel.exists():fail('run prepare first to create Stage C selection')
    return dict(pr=pr,dat=dat,plan=plan,selection=selection,runs=runs,oracle=oracle,all_ids=all_ids,b_ids=b_ids,
                remain=remain,csel=csel,out=output,pdoc=pdoc)

def prepare(args):
    c=context(args,False)
    selection=dict(schema='version-shift-v225-compatible-dataset-subset',
                   role='stage_c_pilot_excluded_47',
                   warning='Legacy key stage_b_datasets is required by the v2.2.5 unchanged scheduler; this is NOT a new Stage B sample',
                   source_stage_b_selection_sha256=EXPECTED_B,
                   plan_sha256=EXPECTED_PLAN,
                   stage_b_datasets=c['remain'],
                   stratum_weights_by_dataset={k:1.0 for k in c['remain']},
                   excluded_pilot_datasets=sorted(c['b_ids']))
    write_once(c['csel'],selection)
    manifest=dict(project='VersionShift v2.2.5',action='CPU completion, no GPU',
                  plan_sha256=EXPECTED_PLAN,stage_b_selection_sha256=EXPECTED_B,
                  stage_c47_selection_sha256=sha(c['csel']),oracle_frame_sha256=EXPECTED_ORACLE,
                  total_eligible=63,stage_b_seen=16,unseen_stage_c=47,
                  original_stage_c_floor=48,
                  interpretation='63-dataset finite-frame census includes observed 16; 47-dataset sensitivity is pilot-excluded and one below K floor',
                  external_timestamp_evidence='NOT_VERIFIED_BY_THIS_SCRIPT',
                  author_confirmation='NOT_VERIFIED_BY_THIS_SCRIPT',
                  original_analysis_code_sha256=sha(c['pr']/'code/analysis_v221.py'),
                  original_worker_sha256=sha(c['pr']/'crossversion/worker.py'))
    write_once(c['dat']/'STAGE_C_COMPLETION_PROTOCOL_RECORD.json',manifest)
    print('PREPARED 47 unobserved datasets; selection SHA256:',sha(c['csel']))
    print('No change to PLAN, Stage B selection, frozen oracle, or original source')


def need_run_specs(c, include_aa=True):
    out=[]
    if include_aa:
        for v in AA_VERSIONS:
            py=str(Path.home()/'versionshift_envs'/('sk-'+v)/'bin/python')
            out.append(dict(name='AA_'+v.replace('.','_'), selection=c['selection'],
                            aa=True, releases=[('AA1',py),('AA2',py)],policies='D',pin_from=None))
    for code,old,new,bname in EDGES:
        releases=[(old,str(Path.home()/'versionshift_envs'/('sk-'+old)/'bin/python')),
                  (new,str(Path.home()/'versionshift_envs'/('sk-'+new)/'bin/python'))]
        out.append(dict(name='STAGE_C47_'+code,selection=c['csel'],aa=False,
                        releases=releases,policies='D,E,P,S',pin_from=old))
        out.append(dict(name='STAGE_C47_'+code+'_REVERSE',selection=c['csel'],aa=False,
                        releases=releases,policies='P',pin_from=new))
    return out


def ensure_original_stageb(c):
    for _code,old,new,forward in EDGES:
        for name,pols,pin in ((forward,{'D','E','P','S'},old),(forward+'_REVERSE',{'P'},new)):
            root=c['runs']/name
            for f in ('RUN_LOCK.json','COLLECT_SUMMARY.json','CROSS_RELEASE_FOLDS.csv.gz'):
                if not (root/f).is_file():fail('Stage B evidence missing: '+str(root/f))
            lock=load(root/'RUN_LOCK.json'); summ=load(root/'COLLECT_SUMMARY.json')
            if lock['plan_sha256']!=EXPECTED_PLAN or lock.get('selection_sha256')!=EXPECTED_B:
                fail('Stage B evidence not bound to frozen dataset plan: '+name)
            if lock.get('worker_sha256') != sha(c['pr']/'crossversion/worker.py') or \
                    lock.get('controller_sha256') != sha(c['pr']/'crossversion/controller.py'):
                fail('Stage B original source hash mismatch: '+name)
            if set(lock['selected_datasets'])!=c['b_ids'] or set(lock['policies'])!=pols or lock['pin_from']!=pin:
                fail('Unexpected Stage B run specification: '+name)
            expect=16*2*6*len(pols)
            if summ.get('planned_jobs')!=expect or summ.get('missing_jobs') or sum(summ.get('status_counts',{}).values())!=expect:
                fail('Stage B collection incomplete: '+name)
    print('CHECKED 4 original collected Stage B runs; no source data modified')


def exec_cmd(argv,cwd=None,log=None):
    print('EXEC:', ' '.join(map(str,argv)),flush=True)
    if log:
        Path(log).parent.mkdir(parents=True,exist_ok=True)
        with open(log,'a') as o:
            o.write('\n--- '+time.strftime('%Y-%m-%d %H:%M:%S')+' ---\n')
            o.write(' '.join(map(str,argv))+'\n');o.flush()
            p=subprocess.run(argv,cwd=cwd,stdout=o,stderr=subprocess.STDOUT)
            if p.returncode:fail('command failed with exit code %s; inspect %s' %(p.returncode,log))
    else:
        subprocess.run(argv,cwd=cwd,check=True)


def run(args):
    c=context(args)
    ensure_original_stageb(c)
    launcher=c['pr']/'versionshift_parallel_10_aa.py'
    if not launcher.is_file():fail('copy versionshift_parallel_10_aa.py to project root')
    if not 1<=args.workers<=10:fail('1 to 10 workers only')
    for spec in need_run_specs(c, include_aa=(not args.skip_aa)):
        target=c['runs']/spec['name']
        cmd=[sys.executable,'-u',str(launcher),'--plan-dir',str(c['plan']),
             '--out',str(target),'--workers',str(args.workers)]
        if target.exists():
            if not (target/'RUN_LOCK.json').is_file():
                fail('output directory exists but no lock; preserve and inspect: '+str(target))
            print('RESUME EXISTING',spec['name'])
        else:
            cmd+=['--init','--selection',str(spec['selection']),'--policies',spec['policies']]
            for rel,py in spec['releases']:cmd+=['--release',rel+'='+py]
            if spec['pin_from']:cmd+=['--pin-from',spec['pin_from']]
            if spec['aa']:cmd+=['--aa']
        exec_cmd(cmd,cwd=c['pr'],log=c['dat']/('CPU_'+spec['name']+'.log'))
        exec_cmd([sys.executable,str(c['pr']/'crossversion/controller.py'),'collect',
                  '--plan-dir',str(c['plan']),'--out',str(target)],cwd=c['pr'],log=c['dat']/('CPU_'+spec['name']+'.log'))
        exec_cmd([sys.executable,str(c['pr']/'crossversion/controller.py'),'replay',
                  '--plan-dir',str(c['plan']),'--out',str(target)],cwd=c['pr'],log=c['dat']/('CPU_'+spec['name']+'.log'))
        if spec['aa']:
            verify_aa(target,c['plan'],c['b_ids'])
        print('FINISHED',spec['name'],flush=True)
    print('CPU FITTING AND REPLAY FINISHED. Next: analyze')


def verify_aa(root,plan,selected):
    import numpy as np
    records=root/'results'; checked=0
    for dataset in sorted(selected):
        for algo in ALGOS:
            left=records/f'AA1__{dataset}__{algo}__D.json'
            right=records/f'AA2__{dataset}__{algo}__D.json'
            if not left.exists() or not right.exists():fail('missing A/A job result '+dataset+' '+algo)
            a,b=load(left),load(right)
            if a.get('status')!='ok' or b.get('status')!='ok':fail('non-OK A/A job '+dataset+' '+algo)
            if a['fold_keys']!=b['fold_keys']:fail('different A/A fold identities')
            with np.load(left.with_suffix('.npz'),allow_pickle=False) as za, np.load(right.with_suffix('.npz'),allow_pickle=False) as zb:
                if set(za.files)!=set(zb.files):fail('A/A prediction keys differ')
                if any(not np.array_equal(za[k],zb[k]) for k in za.files):fail('A/A predictions differ for '+dataset+' '+algo)
            checked+=1
    if checked!=len(selected)*len(ALGOS):fail('incomplete A/A job count')
    (root/'AA_IDENTITY.json').write_text(json.dumps(dict(status='PASS',pairs=checked,prediction_identity='elementwise_exact'),indent=2))
    print('A/A BITWISE EQUALITY PASS:',checked,'model job pairs',flush=True)


def verify_runs(c):
    ensure_original_stageb(c)
    expected=set(c['all_ids'])
    for _code,old,new,bname in EDGES:
        for name,pols,pin in (('STAGE_C47_'+_code,{'D','E','P','S'},old),('STAGE_C47_'+_code+'_REVERSE',{'P'},new)):
            root=c['runs']/name
            for fn in ('RUN_LOCK.json','COLLECT_SUMMARY.json','CROSS_RELEASE_FOLDS.csv.gz'):
                if not (root/fn).exists():fail('not yet completed: '+str(root/fn))
            lock=load(root/'RUN_LOCK.json');summ=load(root/'COLLECT_SUMMARY.json')
            if lock['plan_sha256']!=EXPECTED_PLAN or lock.get('selection_sha256')!=sha(c['csel']):fail('Stage C run lock wrong '+name)
            if set(lock['selected_datasets'])!=set(c['remain']):fail('Stage C 47 selection wrong '+name)
            if set(lock['policies'])!=pols or lock['pin_from']!=pin:fail('Stage C policies/pin wrong '+name)
            expect=47*2*6*len(pols)
            if summ.get('planned_jobs')!=expect or summ.get('missing_jobs') or sum(summ.get('status_counts',{}).values())!=expect:
                fail('Stage C 47 missing or extra jobs in '+name)
    print('STAGE C 47: all 4 run inventories present, locked, collected')


def combine(c):
    import pandas as pd
    merge_dir=c['out']/'merged';merge_dir.mkdir(parents=True,exist_ok=True)
    result={}
    for code,old,new,bname in EDGES:
        for reverse in (False,True):
            prev=bname+('_REVERSE' if reverse else '')
            later='STAGE_C47_'+code+('_REVERSE' if reverse else '')
            items=[]
            for rn in (prev,later):
                p=c['runs']/rn/'CROSS_RELEASE_FOLDS.csv.gz'
                df=pd.read_csv(p)
                ds=set(df['dataset'].astype(str))
                if ds != (c['b_ids'] if rn==prev else set(c['remain'])):
                    fail('dataset coverage mismatch in '+rn)
                items.append(df)
            frame=pd.concat(items,ignore_index=True)
            if set(frame.dataset.astype(str))!=c['all_ids']:fail('combined missing datasets')
            dup=frame[frame.status.eq('ok')].duplicated(['dataset','release','algo','policy','rep','fold'])
            if dup.any():fail('duplicate fold identity in combined frame '+code)
            if frame.status.eq('ok').sum()==0:fail('no good fold rows')
            dest=merge_dir/(code+('_REVERSE' if reverse else '')+'.csv.gz')
            # Derived outputs may be rebuilt; compare if already written to avoid silent overwrite.
            if dest.exists():
                frame.to_csv(dest.with_name(dest.name+'.tmp'),index=False,compression='gzip')
                if not pd.read_csv(dest.with_name(dest.name+'.tmp')).equals(pd.read_csv(dest)):
                    fail('derived CSV changed; preserve existing: '+str(dest))
                dest.with_name(dest.name+'.tmp').unlink()
            else:
                frame.to_csv(dest,index=False,compression='gzip')
            result[code+('_REVERSE' if reverse else '')]=dest
    return result


def create_cell_c(forward,reverse,old,out):
    import pandas as pd
    f=pd.read_csv(forward);r=pd.read_csv(reverse)
    old_D=f[f.release.eq(old)&f.policy.eq('D')]
    old_P=r[r.release.eq(old)&r.policy.eq('P')]
    if old_D.empty or old_P.empty:fail('not enough reverse pinning records for '+old)
    z=pd.concat([old_D,old_P],ignore_index=True)
    if set(z.dataset.astype(str))!=set(f.dataset.astype(str)):fail('Cell-C dataset set differs from full frame')
    if z[z.status.eq('ok')].duplicated(['dataset','release','policy','algo','rep','fold']).any():fail('Cell-C duplicates')
    z.to_csv(out,index=False,compression='gzip');return out


def analytical_commands(c,merged):
    pr=c['pr'];out=c['out'];o=out/'reports';o.mkdir(parents=True,exist_ok=True)
    an=pr/'code/analysis_v221.py'
    pos,neg=merged['POS'],merged['NEG']
    common=['--plan-dir',str(c['plan']),'--oracle-frozen',str(c['oracle'])]
    def analysis(cmd,outname):
        dest=o/outname
        if dest.exists():
            print('SKIP existing analysis',outname);return
        exec_cmd([sys.executable,str(an),*cmd,*common,'--out',str(dest)],cwd=pr,log=out/'ANALYSIS_EXECUTION.log')
    edge=['edge','--inp',str(pos),str(neg),'--edges','0.21.3:0.22.2.post1,1.8.0:1.9.1',
          '--primary','0.21.3:0.22.2.post1,1.8.0:1.9.1']
    analysis(edge+['--population','frame_census'],'FULL63_FRAME_CENSUS.json')
    # The 47 previously unseen tasks are the complement, not a second random sample.
    # Treat this as a finite-subframe sensitivity, not an independent 63-dataset confirmation.
    raw_dir=out/'merged'
    import pandas as pd
    held=[]
    for src,code in ((pos,'POS'),(neg,'NEG')):
        df=pd.read_csv(src); x=df[df.dataset.astype(str).isin(c['remain'])]
        p=raw_dir/('HOLDOUT47_'+code+'.csv.gz'); x.to_csv(p,index=False,compression='gzip');held.append(p)
    h_edge=['edge','--inp',*[str(p) for p in held],'--edges','0.21.3:0.22.2.post1,1.8.0:1.9.1',
          '--primary','0.21.3:0.22.2.post1,1.8.0:1.9.1','--population','frame_census','--selection',str(c['csel'])]
    analysis(h_edge,'HOLDOUT47_FINITE_SUBFRAME_SENSITIVITY.json')
    for code,old,new,_ in EDGES:
        source=merged[code]
        analysis(['mitigation','--inp',str(source),'--edge',old+':'+new,'--no-pt'],
                 'MITIGATION_'+code+'_FULL63.json')
    # Stage-B-first sequential wealth path: descriptive only. Stage C was planned after
    # observed pilot results; conditional exchangeability is NOT automatically guaranteed.
    import numpy as np
    sys.path.insert(0,str(pr/'code'))
    import analysis_v221 as AN
    from numpy.random import default_rng
    for code,old,new,_ in EDGES:
        data=pd.read_csv(merged[code]);ok=AN.validate(data,6,5)
        S,_,ds=AN.shift_matrix(ok,old,new,'D','D')
        Mk=pd.Series(np.abs(S.to_numpy()).mean(axis=1),index=ds)
        if set(Mk.index)!=c['all_ids'] or Mk.isna().any():
            write_once(o/('LENS_D_STAGEB_FIRST_'+code+'_EXPLORATORY.json'),
                       dict(status='INCOMPLETE_CASES',warning='Not calculated because not every dataset has a complete pair-shift matrix'))
            continue
        rng=default_rng(20261106)
        # Two independent frozen permutations; pilot must enter before its complement.
        first=rng.permutation(sorted(c['b_ids'])).tolist()
        last=rng.permutation(sorted(c['remain'])).tolist()
        order=first+last; wealth=1.0;mean=0.5;var=0.25;trace=[];stop=None
        for i,dsid in enumerate(order):
            x=float(Mk[dsid]);delta=mean-0.5
            lam=float(np.clip(delta/(var+delta*delta),0.0,1.0)) if var+delta*delta>0 else 0.0
            wealth*=1.0+lam*(x-0.5)
            if stop is None and wealth>=20:stop=i+1
            n=i+1;dev=x-mean
            mean=(n*mean+x)/(n+1);var=(n*var+dev*dev)/(n+1)
            trace.append(dict(dataset=dsid,position=n,phase='pilot' if i<16 else 'pilot_excluded',M_pp=x,wealth=float(wealth)))
        write_once(o/('LENS_D_STAGEB_FIRST_'+code+'_EXPLORATORY.json'),dict(
              role='DESCRIPTIVE sequential wealth, not a new valid preregistered inference claim',
              warning='Stage C sample and analysis follow observed Stage B results; conditional-mean/exchangeability assumptions need independent justification',
              margin_pp=0.5,order_seed=20261106,alpha=0.05,wealth_final=float(wealth),first_crossing_20=stop,trace=trace))
    p=pr/'versionshift_policy_descriptives.py'
    if not p.is_file():fail('copy versionshift_policy_descriptives.py to project root')
    d=o/'POLICIES_FULL63_DESCRIPTIVE.json'
    if not d.exists():exec_cmd([sys.executable,str(p),'--project',str(pr),'--positive',str(pos),
                                '--later',str(neg),'--out',str(d)],cwd=pr,log=out/'ANALYSIS_EXECUTION.log')
    for code,old,new,_ in EDGES:
        source=merged[code];cfile=raw_dir/('CELL_C_'+code+'.csv.gz')
        if not cfile.exists():create_cell_c(source,merged[code+'_REVERSE'],old,cfile)
        analysis(['attribution',
                  '--cell','A='+str(source)+'@'+old+':D',
                  '--cell','B='+str(source)+'@'+new+':P',
                  '--cell','C='+str(cfile)+'@'+old+':P',
                  '--cell','D='+str(source)+'@'+new+':D'],
                 'ATTRIBUTION_'+code+'_FULL63.json')
    for code,old,new,_ in EDGES:
        for policy in ('D','S'):
            dest=o/('LENSES_'+code+'_'+policy+'_EXPLORATORY.json')
            if dest.exists():continue
            # Secondary lenses do not change any frozen primary inference.
            exec_cmd([sys.executable,str(pr/'code/lenses_crossrelease_v225.py'),
                     '--inp',str(merged[code]),'--edge',old+':'+new,'--policy',policy,
                     '--plan-dir',str(c['plan']),'--oracle-frozen',str(c['oracle']),
                     '--out',str(dest)],cwd=pr,log=out/'ANALYSIS_EXECUTION.log')
    return o


def emulate_fidelity(c,merged):
    import numpy as np, pandas as pd
    pr=c['pr'];o=c['out']/'reports'
    emufile=c['out']/'EMULATED_STAGE_B16_R21_R19.csv'
    if not emufile.exists():
        args=['--plan-dir',str(c['plan']),'--out',str(emufile),'--datasets',','.join(sorted(c['b_ids']))]
        exec_cmd([sys.executable,str(pr/'code/emulate_plan.py'),*args],cwd=pr,log=c['out']/'EMULATION_EXECUTION.log')
    sys.path.insert(0,str(pr/'code'))
    import analysis_v221 as av
    # Compare pairwise changes for original Stage B 16 only, avoiding reuse of Stage C seen datasets.
    real=pd.read_csv(c['runs']/'EDGE_0213_0222'/'CROSS_RELEASE_FOLDS.csv.gz')
    emu=pd.read_csv(emufile)
    ar=av.validate(real,6,5);ae=av.validate(emu,6,5)
    sr,_,dr=av.shift_matrix(ar,'0.21.3','0.22.2.post1','D','D')
    se,_,de=av.shift_matrix(ae,'emu_R21','emu_R19','D','D')
    sr=sr.sort_index().sort_index(axis=1);se=se.reindex(index=sr.index,columns=sr.columns)
    vr=sr.to_numpy().ravel(); ve=se.to_numpy().ravel()
    valid=np.isfinite(vr)&np.isfinite(ve)
    corr=float(np.corrcoef(vr[valid],ve[valid])[0,1]) if valid.sum()>2 and np.std(ve[valid])>0 and np.std(vr[valid])>0 else None
    error=float(np.mean(np.abs(vr[valid]-ve[valid]))) if valid.sum() else None
    rec=dict(role='G5 exploratory emulation-fidelity diagnostic',
             caveat='emu_R19 uses 1.9-era defaults, whereas the real observed comparator is 0.22.2.post1; not perfectly matched counterfactual endpoints',
             k_datasets=len(sr),n_pair_deltas=int(valid.sum()),pearson_pair_delta=corr,
             mean_absolute_pair_delta_difference_pp=error,
             real_M_pp=float(np.nanmean(np.abs(sr.to_numpy()))),
             emulator_M_pp=float(np.nanmean(np.abs(se.to_numpy()))))
    write_once(o/'EMULATION_FIDELITY_STAGE_B16.json',rec)
    return rec


def assess(args):
    c=context(args)
    verify_runs(c)
    for version in AA_VERSIONS:
        root=c['runs']/('AA_'+version.replace('.','_'))
        if not (root/'AA_IDENTITY.json').exists() or load(root/'AA_IDENTITY.json').get('status')!='PASS':
            fail('G1 A/A control not complete for '+version)
    merged=combine(c)
    o=analytical_commands(c,merged)
    fidelity=emulate_fidelity(c,merged) if not args.skip_emulation else {'status':'NOT_PERFORMED'}
    primary=load(o/'FULL63_FRAME_CENSUS.json')
    holdout=load(o/'HOLDOUT47_FINITE_SUBFRAME_SENSITIVITY.json')
    statuses={}
    for code,old,new,_ in EDGES:
        r=primary['margins']['0.5'][old+':'+new]
        h=holdout['margins']['0.5'][old+':'+new]
        statuses[code]=dict(full_63_M_pp=r['q_mean']['M_pp'],full_63_labels=r['labels'],
                            full_63_complete_cases=r['K_complete_case'],
                            holdout_47_M_pp=h['q_mean']['M_pp'],holdout_47_labels=h['labels'],
                            holdout_47_complete_cases=h['K_complete_case'],
                            full_63_frame_affected_share=r['q_prev']['primary_avg_pair']['estimate'])
    report=dict(status='CPU_COMPUTATION_COMPLETE_WITH_SCIENTIFIC_QUALIFICATIONS',
                immutable_inputs={'plan_sha256':EXPECTED_PLAN,'stage_b_selection_sha256':EXPECTED_B,
                                  'oracle_frame_sha256':EXPECTED_ORACLE,'stage_c47_selection_sha256':sha(c['csel'])},
                datasets={'eligible':63,'pilot_seen':16,'previously_unseen':47,'floor':48},
                primary_results=statuses,emulation_fidelity=fidelity,
                gates={'G1':'A/A 4 environments passed and replay checked',
                       'G2':'inspect plan_coverage in FULL63_FRAME_CENSUS.json',
                       'G3':'local freeze validated; external immutable timestamp and author approval not verified',
                       'G4':'inspect attribution identifiability, use preregistered single-path fallback if needed',
                       'G5':'exploratory fidelity reported with non-identical emulated endpoints',
                       'G6':'63 whole finite frame; 47 untouched one below 48 floor; pilot reuse flagged',
                       'G7':'analysis_v221 strict validation executed'},
                scientific_status='NOT_UNTOUCHED_CONFIRMATION; A/A and G2/G7 completed computationally; external G3 proof and floor limitation remain',
                note='Stage C finite frame includes 16 Stage B datasets previously examined; 47 untouched complement reported separately')
    write_once(o/'CPU_COMPLETION_ASSESSMENT.json',report)
    hashes={}
    for root in (o,c['out']/'merged'):
        for f in sorted(root.glob('*')):
            if f.is_file():hashes[str(f.relative_to(c['out']))]=sha(f)
    write_once(c['out']/'CPU_OUTPUT_SHA256.json',hashes)
    print(json.dumps(dict(status=report['status'],primary=statuses,scientific_status=report['scientific_status']),indent=2),flush=True)
    print('OUTPUT:',o)


def status(args):
    c=context(args,False)
    print('FROZEN PLAN',EXPECTED_PLAN,'63 eligible')
    print('FROZEN PILOT SELECTION',EXPECTED_B,'16 observed')
    print('FROZEN ORACLE',EXPECTED_ORACLE)
    print('STAGE C DATASETS',len(c['remain']))
    for spec in need_run_specs(c,True):
        root=c['runs']/spec['name']; result=root/'results';
        count=len(list(result.glob('*.json'))) if result.is_dir() else 0
        total=len(spec['releases'])*len(ALGOS)*len((spec['policies'].split(',')))*(16 if spec['aa'] else 47)
        print(f"{spec['name']:<29}: {count}/{total}")


LONG_RELEASES=('0.21.3','0.22.2.post1','0.23.2','0.24.2','1.0.2','1.1.3','1.2.2',
               '1.3.2','1.4.2','1.5.2','1.6.1','1.7.2','1.8.0','1.9.1')

def run_longitudinal(args):
    """Exploratory/coded-edge RQ3/RQ4 path, all 14 historical releases, D+E on pilot-excluded 47."""
    c=context(args);ensure_original_stageb(c)
    if not 1<=args.workers<=10: fail('workers must be 1..10')
    missing=[v for v in LONG_RELEASES if not (Path.home()/'versionshift_envs'/('sk-'+v)/'bin/python').is_file()]
    if missing:
        fail('cannot start all-release scan; missing genuine historical interpreters: '+', '.join(missing)+
             '; create them using unmodified build_envs.sh, and verify locks')
    # Introspect ALL versions before fitting; source-diff metadata are kept separate
    # from the prospectively frozen oracle and never used to relabel outcomes.
    introspection=c['dat']/'STAGE_C_ALL14_INTROSPECTION.json'
    if not introspection.exists():
        cmd=[sys.executable,str(c['pr']/'crossversion/controller.py'),'introspect']
        for v in LONG_RELEASES:
            cmd+=['--release',v+'='+str(Path.home()/'versionshift_envs'/('sk-'+v)/'bin/python')]
        cmd+=['--out',str(introspection)]
        exec_cmd(cmd,cwd=c['pr'],log=c['dat']/'LONGITUDINAL_INTROSPECTION.log')
    check=load(introspection)
    if set(check['releases'])!=set(LONG_RELEASES):fail('all-14 introspection incomplete')
    print('ALL-RELEASE INTROSPECTION SHA256:',sha(introspection))
    root=c['runs']/'LONGITUDINAL14_D_E_HOLDOUT47'
    launcher=c['pr']/'versionshift_parallel_10_aa.py'
    cmd=[sys.executable,'-u',str(launcher),'--plan-dir',str(c['plan']),'--out',str(root),'--workers',str(args.workers)]
    if not root.exists():
        cmd+=['--init','--selection',str(c['csel']),'--policies','D,E']
        for v in LONG_RELEASES:
            cmd+=['--release',v+'='+str(Path.home()/'versionshift_envs'/('sk-'+v)/'bin/python')]
    elif not (root/'RUN_LOCK.json').exists():
        fail('longitudinal run output exists without valid lock: '+str(root))
    exec_cmd(cmd,cwd=c['pr'],log=c['dat']/'LONGITUDINAL_CPU.log')
    for action in ('collect','replay'):
        exec_cmd([sys.executable,str(c['pr']/'crossversion/controller.py'),action,
                  '--plan-dir',str(c['plan']),'--out',str(root)],cwd=c['pr'],log=c['dat']/'LONGITUDINAL_CPU.log')
    print('LONGITUDINAL 14-RELEASE CPU COLLECTION AND REPLAY COMPLETE')


def analyze_longitudinal(args):
    import pandas as pd
    c=context(args);root=c['runs']/'LONGITUDINAL14_D_E_HOLDOUT47';out=c['out']/'longitudinal'
    f=root/'CROSS_RELEASE_FOLDS.csv.gz'
    if not f.exists():fail('longitudinal run not collected')
    lock=load(root/'RUN_LOCK.json');summary=load(root/'COLLECT_SUMMARY.json')
    if set(lock['releases'])!=set(LONG_RELEASES) or set(lock['policies'])!={'D','E'} or set(lock['selected_datasets'])!=set(c['remain']):
        fail('longitudinal release/policy/dataset lock mismatch')
    if lock['plan_sha256']!=EXPECTED_PLAN or lock['selection_sha256']!=sha(c['csel']):fail('longitudinal lock hash mismatch')
    if summary['planned_jobs']!=len(c['remain'])*len(LONG_RELEASES)*len(ALGOS)*2 or summary.get('missing_jobs'):
        fail('longitudinal missing jobs')
    out.mkdir(parents=True,exist_ok=True)
    rows=pd.read_csv(f)
    if set(rows.dataset.astype(str))!=set(c['remain']):fail('longitudinal dataset frame mismatch')
    # Frame-census describes only the 47-dataset untouched subset; secondary longitudinal family.
    edges=[a+':'+b for a,b in zip(LONG_RELEASES,LONG_RELEASES[1:])]
    report=out/'LONGITUDINAL_EDGE_DECISIONS_EXPLORATORY.json'
    if not report.exists():
        exec_cmd([sys.executable,str(c['pr']/'code/analysis_v221.py'),'edge',
                  '--inp',str(f),'--edges',','.join(edges),'--population','frame_census',
                  '--primary','0.21.3:0.22.2.post1,1.8.0:1.9.1',
                  '--selection',str(c['csel']),'--plan-dir',str(c['plan']),
                  '--oracle-frozen',str(c['oracle']),'--out',str(report)],cwd=c['pr'],log=out/'LONGITUDINAL_ANALYSIS.log')
    # Changelog predictive value of official release notes. Do not confuse hash-differences with score shifts.
    oracle=pd.read_csv(c['pr']/'ORACLE_FRAME_REVIEWED.csv',keep_default_na=False)
    if len(oracle)!=78:fail('oracle frame not 78 opportunities')
    outcome=[]
    # Each column pred_sha is a digest of label bytes for the EXACT frozen dataset/repetition/fold.
    d=rows[(rows.policy=='D')&(rows.status=='ok')].copy()
    d['fold_key']=d.rep.astype(str)+'_'+d.fold.astype(str)
    by={(str(k[0]),str(k[1]),str(k[2])):v for k,v in d.groupby(['dataset','release','algo'])}
    for a,b in zip(LONG_RELEASES,LONG_RELEASES[1:]):
        for algo in ALGOS:
            n_usable=0;n_changed=0;n_folds=0;n_folds_diff=0;n_score_gt=0
            for ds in sorted(c['remain']):
                old=by.get((ds,a,algo));new=by.get((ds,b,algo))
                if old is None or new is None or len(old)!=30 or len(new)!=30:continue
                om=old.set_index('fold_key'); nm=new.set_index('fold_key')
                if set(om.index)!=set(nm.index):fail('non-identical folds in longitudinal edge')
                overlap=sorted(om.index)
                changed=sum(str(om.loc[k,'pred_sha'])!=str(nm.loc[k,'pred_sha']) for k in overlap)
                score_shift=abs(float(om.ba.mean()-nm.ba.mean()))*100
                n_folds+=len(overlap);n_folds_diff+=changed;n_usable+=1
                n_changed+=int(changed>0);n_score_gt+=int(score_shift>0.5)
            if n_usable==0:status='NOT_ESTIMABLE'
            else:status='MEASURED'
            outcome.append(dict(edge=a+'->'+b,estimator=algo,n_datasets_evaluable=n_usable,
                                datasets_with_prediction_change=n_changed,
                                proportion_datasets_prediction_change=n_changed/n_usable if n_usable else None,
                                datasets_score_shift_gt_0_5_pp=n_score_gt,
                                changed_fold_prediction_hashes=n_folds_diff,compared_folds=n_folds,
                                status=status))
    o=pd.DataFrame(outcome)
    joined=oracle.merge(o,on=['edge','estimator'],how='outer',validate='one_to_one',indicator=True)
    if len(joined)!=78 or not joined._merge.eq('both').all():fail('oracle and 78 observed opportunities do not align')
    joined=joined.drop(columns='_merge')
    joined.to_csv(out/'LONGITUDINAL_CHANGELOG_PREDICTIONS.csv',index=False)
    from collections import Counter
    evaluations={}
    for rule,categories in [('strict',{'change_expected','change_possible'}),
                            ('lenient',{'change_expected','change_possible','fp_level_change_possible'})]:
        valid=joined[joined.n_datasets_evaluable.gt(0)].copy()
        predicted=valid.predicted_impact_category.isin(categories)
        actual=valid.datasets_with_prediction_change.gt(0)
        tp=int((predicted&actual).sum());fp=int((predicted&~actual).sum());fn=int((~predicted&actual).sum());tn=int((~predicted&~actual).sum())
        evaluations[rule]=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else None,
                                recall=tp/(tp+fn) if tp+fn else None,valid_opportunities=len(valid),
                                trigger='at least one frozen fold prediction-hash change on one of 47 datasets')
    write_once(out/'CHANGELOG_ORACLE_PRECISION_RECALL.json',dict(population='pilot-excluded 47 CC18 datasets',
              evidence='historical D predictions',note='strict and lenient positive definitions; descriptive and exploratory',
              excluded_opportunities=int(joined.n_datasets_evaluable.eq(0).sum()),measures=evaluations))
    print('LONGITUDINAL RQ3/RQ4 analysis:',out)


def run_fixedstack(args):
    c=context(args)
    releases=[('1.4.2',str(Path.home()/'versionshift_envs/sk-1.4.2-fixedstack/bin/python')),
              ('1.5.2',str(Path.home()/'versionshift_envs/sk-1.5.2-fixedstack/bin/python'))]
    if any(not Path(py).is_file() for _,py in releases):fail('fixed-stack environment(s) not built')
    name='FIXEDSTACK_142_152_D_E_HOLDOUT47';root=c['runs']/name
    cmd=[sys.executable,'-u',str(c['pr']/'versionshift_parallel_10_aa.py'),
         '--plan-dir',str(c['plan']),'--out',str(root),'--workers',str(args.workers)]
    if not root.exists():
        cmd+=['--init','--selection',str(c['csel']),'--policies','D,E']
        for v,py in releases:cmd+=['--release',v+'='+py]
    elif not (root/'RUN_LOCK.json').exists():fail('fixedstack run exists but has no lock')
    exec_cmd(cmd,cwd=c['pr'],log=c['dat']/'FIXEDSTACK_CPU.log')
    for action in ('collect','replay'):
        exec_cmd([sys.executable,str(c['pr']/'crossversion/controller.py'),action,
                  '--plan-dir',str(c['plan']),'--out',str(root)],cwd=c['pr'],log=c['dat']/'FIXEDSTACK_CPU.log')
    print('FIXEDSTACK CONTROL COMPLETED:',root)


def analyze_fixedstack(args):
    import pandas as pd
    c=context(args);root=c['runs']/'FIXEDSTACK_142_152_D_E_HOLDOUT47'
    f=root/'CROSS_RELEASE_FOLDS.csv.gz'
    if not f.exists():fail('fixedstack run not collected')
    out=c['out']/'longitudinal';out.mkdir(parents=True,exist_ok=True)
    p=out/'FIXEDSTACK_142_152_EXPLORATORY.json'
    if not p.exists():
        exec_cmd([sys.executable,str(c['pr']/'code/analysis_v221.py'),'edge',
              '--inp',str(f),'--edges','1.4.2:1.5.2','--population','frame_census',
              '--selection',str(c['csel']),'--plan-dir',str(c['plan']),
              '--oracle-frozen',str(c['oracle']),'--out',str(p)],cwd=c['pr'],log=out/'FIXEDSTACK_ANALYSIS.log')
    print('Fixed-stack experimental result:',p)

def main():
    ap=argparse.ArgumentParser(description='CPU-only VersionShift experiment completion (no GPU)')
    ap.add_argument('command',choices=['prepare','run','analyze','status','longitudinal','analyze-long','fixedstack','analyze-fixed'])
    ap.add_argument('--project',default='/mnt/c/VersionShift_v2_2_5')
    ap.add_argument('--data',default=str(Path.home()/'versionshift_data'))
    ap.add_argument('--workers',type=int,default=10)
    ap.add_argument('--skip-aa',action='store_true',help='Only for separately authenticated G1 A/A evidence; analysis still requires A/A files')
    ap.add_argument('--skip-emulation',action='store_true',help='Do not compute G5; marks it incomplete')
    args=ap.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='run':run(args)
    elif args.command=='analyze':assess(args)
    elif args.command=='longitudinal':run_longitudinal(args)
    elif args.command=='analyze-long':analyze_longitudinal(args)
    elif args.command=='fixedstack':run_fixedstack(args)
    elif args.command=='analyze-fixed':analyze_fixedstack(args)
    else:status(args)

if __name__=='__main__':main()
