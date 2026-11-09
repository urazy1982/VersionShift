"""Offline structural regression tests; no historical environments/data required."""
import importlib.util, json, tempfile, hashlib
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('vsfinish',ROOT/'versionshift_cpu_finish.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def make_fixture(tmp):
    pr=tmp/'project';data=tmp/'vdata';pr.mkdir();data.mkdir()
    (pr/'crossversion').mkdir()
    (pr/'crossversion/controller.py').write_text('#stub')
    (pr/'crossversion/worker.py').write_text('#stub')
    (pr/'code').mkdir();(pr/'code/analysis_v221.py').write_text('#stub')
    frame=pr/'ORACLE_FRAME_REVIEWED.csv';frame.write_text('id,status\n1,complete\n')
    m.EXPECTED_ORACLE=m.sha(frame)
    (pr/'ORACLE_FROZEN.json').write_text(json.dumps(dict(status='FROZEN_COMPLETE',frame_sha256=m.EXPECTED_ORACLE,frame_path=str(frame))))
    plan=data/'STAGE_B_PLAN_CACHED';plan.mkdir()
    ids=[f'openml_{i}' for i in range(63)]
    doc=dict(datasets=[dict(name=i) for i in ids])
    (plan/'PLAN.json').write_text(json.dumps(doc))
    m.EXPECTED_PLAN=m.sha(plan/'PLAN.json')
    sel=data/'STAGE_B_SELECTION.json'
    sel.write_text(json.dumps(dict(stage_b_datasets=ids[:16],plan_sha256=m.EXPECTED_PLAN)))
    m.EXPECTED_B=m.sha(sel)
    return SimpleNamespace(project=str(pr),data=str(data)),pr,data,ids

def test_prepare_idempotent(tmp_path):
    args,pr,data,ids=make_fixture(tmp_path)
    m.prepare(args);m.prepare(args)
    c=m.context(args)
    j=m.load(c['csel'])
    assert len(j['stage_b_datasets'])==47
    assert not set(j['stage_b_datasets']) & set(ids[:16])
    assert set(j['stage_b_datasets'])==set(ids[16:])
    assert j['source_stage_b_selection_sha256']==m.EXPECTED_B
    assert (data/'STAGE_C_COMPLETION_PROTOCOL_RECORD.json').exists()

def test_selection_change_fail_closed(tmp_path):
    args,pr,data,ids=make_fixture(tmp_path)
    m.prepare(args)
    sel=data/'STAGE_C_HOLDOUT47_SELECTION.json'
    x=m.load(sel);x['stage_b_datasets'].append(ids[0]);sel.write_text(json.dumps(x))
    try:m.prepare(args)
    except SystemExit as e:assert 'STOP' in str(e)
    else:assert False,'tampered selection must be rejected'

def test_merge_no_cross_dataset_leakage(tmp_path):
    args,pr,data,ids=make_fixture(tmp_path)
    m.prepare(args);c=m.context(args)
    runs=data/'STAGE_B_RUNS';runs.mkdir()
    for code,old,new,bname in m.EDGES:
        for rev in (False,True):
            for name,subids in ((bname+('_REVERSE' if rev else ''), ids[:16]),
                                ('STAGE_C47_'+code+('_REVERSE' if rev else ''),ids[16:])):
                root=runs/name;root.mkdir(); rows=[]
                pols=('P',) if rev else ('D','E','P','S')
                for ds in subids:
                    for rel in (old,new):
                        for algo in m.ALGOS:
                            for pol in pols:
                                rows.append(dict(dataset=ds,release=rel,algo=algo,policy=pol,rep=0,fold=0,
                                                 status='ok',ba=0.5))
                df=pd.DataFrame(rows)
                df.to_csv(root/'CROSS_RELEASE_FOLDS.csv.gz',index=False,compression='gzip')
                summ=dict(planned_jobs=len(subids)*2*6*len(pols),missing_jobs=[],status_counts={'ok':len(df)})
                (root/'COLLECT_SUMMARY.json').write_text(json.dumps(summ))
                lock=dict(worker_sha256=m.sha(pr/'crossversion/worker.py'),
                          controller_sha256=m.sha(pr/'crossversion/controller.py'),
                          plan_sha256=m.EXPECTED_PLAN,
                          selection_sha256=m.EXPECTED_B if subids==ids[:16] else m.sha(c['csel']),
                          selected_datasets=subids,policies=list(pols),pin_from=new if rev else old,
                          releases={old:{},new:{}})
                (root/'RUN_LOCK.json').write_text(json.dumps(lock))
    m.verify_runs(c)
    j=m.combine(c)
    assert set(j)=={'POS','NEG','POS_REVERSE','NEG_REVERSE'}
    for p in j.values():
        df=pd.read_csv(p)
        assert len(set(df.dataset))==63
        assert not df[df.status.eq('ok')].duplicated(['dataset','release','algo','policy','rep','fold']).any()

def test_aa_validation_detects_difference(tmp_path):
    args,pr,data,ids=make_fixture(tmp_path)
    root=data/'aa';root.mkdir();(root/'results').mkdir()
    records=root/'results'
    for name in ids[:16]:
        for alg in m.ALGOS:
            for label in ('AA1','AA2'):
                rec=records/f'{label}__{name}__{alg}__D.json'
                rec.write_text(json.dumps(dict(status='ok',fold_keys=['r0_f0'])))
                np.savez(rec.with_suffix('.npz'),r0_f0_pred=np.array([1,2,3]))
    m.verify_aa(root,pr,ids[:16])
    assert m.load(root/'AA_IDENTITY.json')['pairs']==96
    p=records/f'AA2__{ids[0]}__LR__D.npz'
    np.savez(p,r0_f0_pred=np.array([1,2,0]))
    try:m.verify_aa(root,pr,ids[:16])
    except SystemExit as e:assert 'A/A predictions differ' in str(e)
    else:assert False,'A/A mismatch was not caught'
