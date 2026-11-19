"""Build/provenance readiness guard for the required historical environment set."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .utils import read_json,sha256,require

def validate_historical_readiness(root,record=None,environment_root=None):
    root=Path(root).resolve();record=Path(record) if record else root/'runs/plan/HISTORICAL_ENVIRONMENT_READINESS_v305.json'
    envroot=Path(environment_root) if environment_root else Path.home()/'versionshift_envs'
    require(record.is_file(),'historical environment readiness record absent')
    r=read_json(record);expected=read_json(root/'config/E5_EDGES.json')['release_order']
    require(r.get('status')=='ALL_14_BUILD_VALIDATIONS_PASS_AUTHOR_RATIFICATION_PENDING','historical environment restoration incomplete')
    rows=r['environments'];require(len(rows)==14 and [v['release'] for v in rows]==expected,'historical release coverage differs')
    for v in rows:
        require(v['status'] in ('RESTORED_EXACT_ARCHIVED_PACKAGES_VALIDATED','BUILT_EXACT_RELEASE_ERA_CONSTRAINTS_VALIDATED_RECONSTRUCTED_LOCK'),'historical endpoint blocked')
        require(Path(v['environment_path']).resolve()==(envroot/('sk-'+v['release'])).resolve(),'historical environment path differs')
        p=v['provenance'];require(p['sklearn']==v['release'],'historical exact version mismatch')
        require(p['python_version'].startswith(v['required_python_minor']+'.'),'historical interpreter compatibility differs')
        require(sha256(p['python_binary'])==p['python_binary_sha256'],'historical interpreter binary changed')
        require(sha256(Path(v['environment_path'])/'lock.txt')==v['lock_sha256'],'historical environment lock changed')
        require(p['numerical_backends'] and all(b['num_threads']==1 for b in p['numerical_backends']),'historical numerical thread provenance differs')
        for b in p['numerical_backends']:require(sha256(b['filepath'])==b['binary_sha256'],'historical numerical backend binary changed')
    files=r.get('runtime_files_sha256',{})
    require(files,'historical complete runtime file hashes absent')
    def verify(item):
        path,h=item;require(Path(path).is_file() and sha256(path)==h,'historical runtime bytes changed '+path)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(verify,files.items()))
    return {'status':'PASS_BUILD_ONLY_HISTORICAL_READINESS','record_sha256':sha256(record),'runtime_files':len(files),
            'author_ratification_still_required':True,'original_binary_identity_not_claimed':True}
