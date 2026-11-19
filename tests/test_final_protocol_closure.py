"""Targeted final-policy and historical-readiness synthetic fixtures; no fitting."""
import json,shutil
from pathlib import Path
import pytest
from versionshift_v3.utils import sha256
from versionshift_v3.binding import load_settings
from versionshift_v3.statistics import evaluate_edge,ALGOS
from versionshift_v3.historical_envs import validate_historical_readiness
ROOT=Path(__file__).resolve().parents[1]

def test_finite_frame_policy_and_excluded_E3_are_bound(tmp_path):
 (tmp_path/'config').mkdir()
 for n in ['analysis.json','statistical_spec_v305.json']:shutil.copy2(ROOT/'config'/n,tmp_path/'config'/n)
 _,s,_=load_settings(tmp_path);assert s['prevalence_model_justified'] is False
 p=tmp_path/'config/statistical_spec_v305.json';s['E3_confirmatory_status']='ADOPTED';p.write_text(json.dumps(s))
 with pytest.raises(RuntimeError,match='E3'):load_settings(tmp_path)
 s['E3_confirmatory_status']='EXCLUDED_FUTURE_WORK_PROPOSAL_ONLY';s['population_inference_scope']='UNCONDITIONAL';p.write_text(json.dumps(s))
 with pytest.raises(RuntimeError,match='scope'):load_settings(tmp_path)

def test_numeric_primary_definition_unchanged_and_sensitivity_labeled():
 records={(t,v,a):50.+i+(t*.01*(i+1) if v=='new' else 0) for t in range(4) for v in ('old','new') for i,a in enumerate(ALGOS)}
 r=evaluate_edge(records,releases=['old','new'],bootstrap_reps=10)
 assert r['evidence_scope']=='FINITE_ALLOCATED_TASK_FRAME' and r['population_inference_scope']=='IID_TASK_MODEL_SENSITIVITY_ONLY'
 assert r['pair_count']==28 and r['margin_pp']==.5 and r['equivalence']['status']=='INCONCLUSIVE'

def fake_history(root):
 (root/'config').mkdir();shutil.copy2(ROOT/'config/E5_EDGES.json',root/'config/E5_EDGES.json');versions=json.loads((root/'config/E5_EDGES.json').read_text())['release_order'];env=root/'synthetic_envs';rows=[];files={}
 for version in versions:
  folder=env/('sk-'+version);folder.mkdir(parents=True)
  for name in ['lock.txt','python','backend.so']:(folder/name).write_text('SYNTHETIC FIXTURE '+name)
  rows.append({'release':version,'environment_path':str(folder),'status':'RESTORED_EXACT_ARCHIVED_PACKAGES_VALIDATED','required_python_minor':'3.11','lock_sha256':sha256(folder/'lock.txt'),'provenance':{'sklearn':version,'python_version':'3.11.18','python_binary':str(folder/'python'),'python_binary_sha256':sha256(folder/'python'),'numerical_backends':[{'num_threads':1,'filepath':str(folder/'backend.so'),'binary_sha256':sha256(folder/'backend.so')}]}})
  files.update({str(p):sha256(p) for p in folder.iterdir()})
 data={'status':'ALL_14_BUILD_VALIDATIONS_PASS_AUTHOR_RATIFICATION_PENDING','environments':rows,'runtime_files_sha256':files};record=root/'fixture_readiness.json';record.write_text(json.dumps(data));return record,env,data

def test_historical_guard_validates_complete14_synthetic_readiness(tmp_path):
 record,env,_=fake_history(tmp_path);r=validate_historical_readiness(tmp_path,record,env);assert r['runtime_files']==42

@pytest.mark.parametrize('attack',['missing_endpoint','blocked','interpreter','backend','runtime_file','coverage'])
def test_historical_guard_rejects_changes(tmp_path,attack):
 record,env,data=fake_history(tmp_path);v=data['environments'][0]
 if attack=='missing_endpoint':v['environment_path']=str(env/'wrong')
 elif attack=='blocked':v['status']='BLOCKED'
 elif attack=='interpreter':Path(v['provenance']['python_binary']).write_text('changed')
 elif attack=='backend':Path(v['provenance']['numerical_backends'][0]['filepath']).write_text('changed')
 elif attack=='runtime_file':
  p=env/'extra';p.write_text('original');data['runtime_files_sha256'][str(p)]=sha256(p);p.write_text('changed')
 else:data['environments'].pop()
 record.write_text(json.dumps(data))
 with pytest.raises(RuntimeError):validate_historical_readiness(tmp_path,record,env)
