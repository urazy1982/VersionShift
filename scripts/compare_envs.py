import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from versionshift_v3.utils import write_once,read_json
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
r=Path(a.root);vs=['1.4.2','1.5.2','1.6.1'];data=[read_json(r/('preflight-'+v+'.json')) for v in vs]
keys=['python_minor','numpy','scipy']
# BLAS paths include each venv path; compare implementation/num_threads/version instead.
def normalized_threads(v):
    if not isinstance(v,list):return v
    return sorted([{'api':x.get('internal_api'),'thread':x.get('num_threads'),
                    'version':x.get('version'),'prefix':x.get('prefix')} for x in v],
                  key=lambda x:repr(x))
diff={k:[x.get(k) for x in data] for k in keys if len({json.dumps(x.get(k),sort_keys=True,default=str) for x in data})>1}
threads=[normalized_threads(x.get('threadpool_info')) for x in data]
if len({json.dumps(x,sort_keys=True) for x in threads})>1:diff['threadpool_info']=threads
pass_=all(x.get('status')=='PASS' for x in data) and not diff
report={'status':'PASS' if pass_ else 'FAIL','release_versions':vs,'stack_differences':diff}
write_once(a.out,report)
print(json.dumps(report,indent=2))
if not pass_:raise SystemExit(2)
