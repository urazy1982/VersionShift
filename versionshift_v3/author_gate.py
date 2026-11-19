"""Hash-bound separate E2 authorization; never signs or creates author approval.

These checks bind an actual author-supplied record to its signed approval bytes.
They do not cryptographically establish a person's identity or replace independent
human authentication. Synthetic tests use explicitly fictional temporary records.
"""
import datetime,re
from pathlib import Path
from .utils import read_json,sha256,digest_json,require

EXACT_E2_SHA256='18f8c3c00e48c0aa06d3a4c1e299eea73dfbceef1ff5d46957cf1fdff45509f8'
E2_DECISION='RETAIN_EXACT_E2_MODERN_HGB_UNSUPPORTED_NO_IMPUTATION'
REVIEW_FILES=('runs/plan/HISTORICAL_ENVIRONMENT_READINESS_v305.json','versionshift_v3/historical_envs.py','config/E2_legacy_v26.json','config/analysis.json','config/statistical_spec_v305.json',
              'runs/plan/PROSPECTIVE_TASK_ALLOCATION.json','runs/plan/CHANGELOG_ORACLE_16.csv',
              'docs/PROTOCOL_v3.0.2_REVIEWED_DRAFT.md','docs/AMENDMENT_V305.md',
              'versionshift_v3/author_gate.py','versionshift_v3/freeze.py')

def validate_e2_author_authorization(root,author_approval,record=None):
    root=Path(root).resolve();approval=Path(author_approval).resolve()
    require(approval==root/'runs/plan/AUTHOR_APPROVAL.txt','signed approval must be archived at canonical project path')
    path=Path(record) if record is not None else root/'runs/plan/E2_AUTHOR_AUTHORIZATION.json'
    require(path.resolve()==root/'runs/plan/E2_AUTHOR_AUTHORIZATION.json','E2 authorization must use canonical project path')
    require(path.is_file(),'separate actual author E2 authorization absent')
    require(approval.is_file(),'actual signed author approval absent')
    r=read_json(path)
    require(set(r)=={'kind','authorization','authorization_sha256','signed_author_approval_sha256'},'invalid authorization envelope')
    require(r['kind']=='E2_SEPARATE_AUTHOR_AUTHORIZATION_V1','wrong E2 authorization kind')
    a=r['authorization']
    require(set(a)=={'author_identity','review_date','author_authorized','decision','e2_sha256','original_source_sha256','reviewed_files_sha256'},'invalid E2 authorization payload')
    require(a['author_authorized'] is True and a['decision']==E2_DECISION,'actual author E2 decision absent or different')
    name=a['author_identity'];require(isinstance(name,str) and name.strip() and name.strip().lower() not in {'pending','draft','unknown'},'author identity absent')
    require(isinstance(a['review_date'],str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}',a['review_date']) is not None,'actual author review date absent')
    try:datetime.date.fromisoformat(a['review_date'])
    except ValueError:raise RuntimeError('GATE_FAIL: invalid actual author review date')
    e2=root/'config/E2_legacy_v26.json';m=read_json(e2)
    require(sha256(e2)==EXACT_E2_SHA256==a['e2_sha256'],'exact historical E2 bytes changed or authorization differs')
    require(a['original_source_sha256']==m['source_sha256'],'E2 original source authorization differs')
    hashes={p:sha256(root/p) for p in REVIEW_FILES}
    require(a['reviewed_files_sha256']==hashes,'author-reviewed file hashes differ from current inputs')
    payload_hash=digest_json(a);require(r['authorization_sha256']==payload_hash,'authorization payload SHA mismatch')
    require(r['signed_author_approval_sha256']==sha256(approval),'signed author approval bytes differ')
    text=approval.read_text()
    require(re.search(r'^APPROVED\s*$',text,re.MULTILINE) is not None,'signed record lacks explicit APPROVED line')
    require(all(token in text for token in (name,a['review_date'],payload_hash,EXACT_E2_SHA256)),'signed approval does not bind actual identity/date/E2/payload hash')
    return {'record_path':str(path.relative_to(root)),'record_sha256':sha256(path),
            'authorization_sha256':payload_hash,'signed_author_approval_sha256':sha256(approval),
            'identity_authentication':'Actual personal signature/authorization requires independent human authentication; hashes and JSON fields alone do not prove it.'}
