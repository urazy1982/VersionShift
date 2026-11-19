"""Fictional temporary authorization fixtures; never real author approvals."""
import json,shutil
from pathlib import Path
import pytest
from versionshift_v3.author_gate import validate_e2_author_authorization,REVIEW_FILES,EXACT_E2_SHA256,E2_DECISION
from versionshift_v3.utils import sha256,digest_json
ROOT=Path(__file__).resolve().parents[1]

def fictional_fixture(root):
 for n in REVIEW_FILES:
  p=root/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
 m=json.loads((root/'config/E2_legacy_v26.json').read_text())
 a={'author_identity':'SYNTHETIC TEST FIXTURE ONLY','review_date':'2000-01-01','author_authorized':True,'decision':E2_DECISION,'e2_sha256':EXACT_E2_SHA256,'original_source_sha256':m['source_sha256'],'reviewed_files_sha256':{n:sha256(root/n) for n in REVIEW_FILES}}
 h=digest_json(a);approval=root/'runs/plan/AUTHOR_APPROVAL.txt';approval.write_text('SYNTHETIC ENGINEERING FIXTURE; NOT PERSONAL AUTHORIZATION\nAPPROVED\n'+a['author_identity']+'\n'+a['review_date']+'\n'+h+'\n'+EXACT_E2_SHA256+'\n')
 r={'kind':'E2_SEPARATE_AUTHOR_AUTHORIZATION_V1','authorization':a,'authorization_sha256':h,'signed_author_approval_sha256':sha256(approval)}
 record=root/'runs/plan/E2_AUTHOR_AUTHORIZATION.json';record.write_text(json.dumps(r));return record,approval,r

def test_separate_binding_keeps_exact_E2_and_inline_false(tmp_path):
 p,approval,r=fictional_fixture(tmp_path);before=sha256(tmp_path/'config/E2_legacy_v26.json')
 assert json.loads((tmp_path/'config/E2_legacy_v26.json').read_text())['author_confirmed'] is False
 assert validate_e2_author_authorization(tmp_path,approval)['record_sha256']==sha256(p)
 assert sha256(tmp_path/'config/E2_legacy_v26.json')==before==EXACT_E2_SHA256

@pytest.mark.parametrize('attack',['missing','unsigned','hash','decision','date','source','payload','input','inline_edit','NOT_APPROVED'])
def test_missing_or_tampered_author_gates_fail_closed(tmp_path,attack):
 p,approval,r=fictional_fixture(tmp_path)
 if attack=='missing':p.unlink()
 elif attack=='unsigned':approval.unlink()
 elif attack=='hash':r['signed_author_approval_sha256']='0'*64
 elif attack=='decision':r['authorization']['author_authorized']=False
 elif attack=='date':r['authorization']['review_date']='PENDING'
 elif attack=='source':r['authorization']['original_source_sha256']='0'*64
 elif attack=='payload':r['authorization_sha256']='0'*64
 elif attack=='input':(tmp_path/'config/analysis.json').write_text('{}')
 elif attack=='inline_edit':
  e=tmp_path/'config/E2_legacy_v26.json';e.write_text(e.read_text().replace('"author_confirmed": false','"author_confirmed": true'))
 elif attack=='NOT_APPROVED':
  approval.write_text(approval.read_text().replace('\nAPPROVED\n','\nNOT APPROVED\n'));r['signed_author_approval_sha256']=sha256(approval)
 if p.exists():p.write_text(json.dumps(r))
 with pytest.raises(RuntimeError):validate_e2_author_authorization(tmp_path,approval)

def test_freezer_uses_separate_gate_before_any_freeze(tmp_path):
 import inspect
 from versionshift_v3.freeze import freeze
 source=inspect.getsource(freeze)
 assert 'validate_e2_author_authorization(root,author_approval)' in source
 assert "e2.get('author_confirmed') is True" not in source
 assert "'runs/plan/E2_AUTHOR_AUTHORIZATION.json'" in source
 assert "'runs/plan/AUTHOR_APPROVAL.txt'" in source
