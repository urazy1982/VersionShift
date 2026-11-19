"""A package verifier must not follow archived host/data references."""
import csv,json,shutil,sys,subprocess,hashlib
from pathlib import Path
import pytest
from scripts.verify_pre_freeze_package import verify_package
from scripts.verify_pre_freeze_host import verify_host
ROOT=Path(__file__).resolve().parents[1]
def package(root):
 root.mkdir();shutil.copy2(ROOT/'scripts/verify_pre_freeze_package.py',root/'verify_package.py');shutil.copy2(ROOT/'scripts/verify_pre_freeze_host.py',root/'verify_host.py')
 shutil.copy2(ROOT/'reports/FINAL_PRE_FREEZE_20261117/CHANGELOG_16_AUTHOR_DRAFT.csv',root/'CHANGELOG_16_AUTHOR_DRAFT.csv')
 shutil.copy2(ROOT/'reports/AUTHOR_FREEZE_CANDIDATE_20261117/E3_PROPOSAL_NOT_ADOPTED.json',root/'E3_PROPOSAL_NOT_ADOPTED.json')
 (root/'HOST_INVENTORY.json').write_text(json.dumps({'files':[{'host_path':'/does/not/exist/VersionShift','sha256':'0'*64}]}))
 files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
 (root/'PACKAGE_MANIFEST.json').write_text(json.dumps({'scope':'PACKAGE_BYTES_ONLY_NOT_HOST_VERIFICATION','files_sha256':files}))
 return root

def test_fresh_directory_verifier_without_project_imports(tmp_path):
 root=package(tmp_path/'isolated');home=tmp_path/'empty_home';home.mkdir()
 p=subprocess.run([sys.executable,'-I','-S','-B',str(root/'verify_package.py'),'--root',str(root)],cwd=home,env={'HOME':str(home)},capture_output=True,text=True)
 assert p.returncode==0,p.stderr+p.stdout
 assert json.loads(p.stdout)['status']=='PASS_PACKAGE_ONLY' and json.loads(p.stdout)['host_inventory_verified'] is False

@pytest.mark.parametrize('attack',['missing_csv','bad_e3_bytes','escape','extra','symlink'])
def test_corrupted_or_escaping_payload_rejected(tmp_path,attack):
 root=package(tmp_path/'isolated')
 if attack=='missing_csv':(root/'CHANGELOG_16_AUTHOR_DRAFT.csv').unlink()
 elif attack=='bad_e3_bytes':(root/'E3_PROPOSAL_NOT_ADOPTED.json').write_text('{}')
 elif attack=='extra':(root/'unknown.txt').write_text('unbound')
 elif attack=='symlink':(root/'escape_link').symlink_to(tmp_path,target_is_directory=True)
 else:
  p=root/'PACKAGE_MANIFEST.json';m=json.loads(p.read_text());m['files_sha256']['../secret']='0'*64;p.write_text(json.dumps(m))
 with pytest.raises(ValueError):verify_package(root)

def test_host_verifier_requires_actual_external_bytes(tmp_path):
 p=tmp_path/'synthetic_host_data';p.write_text('preserved');h=hashlib.sha256(p.read_bytes()).hexdigest()
 manifest={'scope':'SYNTHETIC_HOST_FIXTURE','files':[{'logical_path':'fixture','host_path':str(p),'sha256':h}]}
 assert verify_host(manifest)['status']=='PASS_HOST_INVENTORY';p.write_text('changed');assert verify_host(manifest)['status']=='FAIL_HOST_INVENTORY'

def test_rehashed_E3_adoption_is_not_accepted_by_proposal_package(tmp_path):
 root=package(tmp_path/'isolated');p=root/'E3_PROPOSAL_NOT_ADOPTED.json';e=json.loads(p.read_text());e['author_confirmed']=True;p.write_text(json.dumps(e))
 m=root/'PACKAGE_MANIFEST.json';v=json.loads(m.read_text());v['files_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest();m.write_text(json.dumps(v))
 with pytest.raises(ValueError,match='proposal-only'):verify_package(root)
