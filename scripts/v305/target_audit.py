"""Audit official task targets without querying model outcomes."""
import sys,json
from pathlib import Path
import openml
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from versionshift_v3.utils import read_json,require,write_once
out=Path(sys.argv[1]);records=[]
for tid in read_json(out/'UNIVERSE_52.json')['task_ids']:
 t=openml.tasks.get_task(tid,download_splits=False,download_data=False);d=openml.datasets.get_dataset(t.dataset_id,download_data=False);r=read_json(out/f'candidates/{tid}/POSTCAP.json')
 require(int(t.dataset_id)==r['dataset_id'],'task source differs');require(d.default_target_attribute in (None,t.target_name),'default and official task target differ')
 require(r.get('target_name',d.default_target_attribute)==t.target_name,'used target differs')
 records.append({'task_id':tid,'dataset_id':int(t.dataset_id),'official_task_target':t.target_name,'dataset_default_target':d.default_target_attribute,'target_used':r.get('target_name',d.default_target_attribute),'match':True})
write_once(out/'TASK_TARGET_AUDIT_52.json',{'status':'PASS','records':records});print('52 official targets verified')
