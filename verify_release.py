import json,hashlib,re,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 manifest=json.loads((ROOT/'MANIFEST_SHA256.json').read_text())
 actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['.git','__pycache__','.pytest_cache','.venv'])}
 assert actual==set(manifest)|{'MANIFEST_SHA256.json'},'unmanifested or missing files'
 for f,h in manifest.items():assert sha(ROOT/f)==h,'payload mismatch '+f
 frozen=ROOT/'runs/plan/FROZEN_PROTOCOL.json';assert sha(frozen)=='7a62fb456f89dc58d2797e0e5ce83a456b178b4e740de93e6e9eacfeeca56f55'
 plan=json.loads((ROOT/'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json').read_text());groups=[plan[k] for k in ['pilot_task_ids','evaluation_task_ids','external_task_ids']]
 assert list(map(len,groups))==[15,48,20] and len(set(sum(groups,[])))==83
 core={x['dataset_id'] for x in plan['original_tasks']};ext=[x['dataset_id'] for x in plan['external_tasks']];assert len(set(ext))==20 and not core&set(ext)
 index=json.loads((ROOT/'runs/plan/MATRICES_INDEX.json').read_text());assert len(index['task_manifests'])==83
 count=0
 for e in index['task_manifests']:
  p=ROOT/e['path'];assert sha(p)==e['sha256'];m=json.loads(p.read_text());assert m['task_id']==e['task_id'];fs=m['folds'];assert len(fs)==60
  assert {(x['key'],x['preprocessing_id']) for x in fs}=={(f'r{r}_f{f}',v) for r in range(6) for f in range(5) for v in ['P1','P2']}
  assert all(re.fullmatch('[0-9a-f]{64}',x['sha256']) and 0<x['n_features']<=500 for x in fs);count+=len(fs)
 assert count==4980
 ready=json.loads((ROOT/'runs/plan/HISTORICAL_ENVIRONMENT_READINESS_v305.json').read_text());assert len(ready['environments'])==14
 for row in ready['environments']:
  lock=(ROOT/'environments/historical'/('sk-'+row['release'])/'lock.txt').read_text()
  assert re.search(r'^scikit-learn=='+re.escape(row['release'])+r'\?$',lock,re.M) or ('scikit-learn=='+row['release']) in lock
  assert '--hash=sha256:' in lock
 e2=json.loads((ROOT/'config/E2_legacy_v26.json').read_text());assert e2['source_sha256']=='ee39adb1d9cda6030e62245dc6e10576e9b38056de94403c2355e3086431d90c'
 assert hashlib.sha256(json.dumps(e2['parameters'],sort_keys=True,separators=(',',':')).encode()).hexdigest()=='37d38861aaddd5f85a379ec08e6cf62372f9657c160a45811e56b2edb7775152'
 source=ROOT/'evidence/v26/G3E_worker.py';assert sha(source)==e2['source_sha256']
 nodes=[n.value for n in ast.parse(source.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXPLICIT' for t in n.targets)];assert len(nodes)==1
 def literal(n):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict' and not n.args:return {k.arg:literal(k.value) for k in n.keywords}
  if isinstance(n,ast.Dict):return {literal(k):literal(v) for k,v in zip(n.keys,n.values)}
  return ast.literal_eval(n)
 assert literal(nodes[0])==e2['parameters'],'public E2 scientific parameters differ from exact original source AST'
 assert sha(ROOT/'requirements-test.lock')==manifest['requirements-test.lock']
 assert not (ROOT/'runs/plan/AUTHOR_APPROVAL.txt').exists()
 print(json.dumps(dict(status='PASS_PUBLIC_PACKAGE_ONLY',files=len(manifest),tasks=83,matrix_commitments=count,historical_environments=14,frozen_sha256=sha(frozen),matrix_payloads_included=False,signature_records_included=False,host_replay_claimed=False)))
if __name__=='__main__':verify()
