"""Verify OpenML ARFF bytes even when OpenML returns an existing cache file."""
import hashlib,os
from pathlib import Path
from .utils import require,sha256

def verify_arff(dataset):
    require(os.environ.get('OPENML_SKIP_PARQUET','').lower()=='true','ARFF integrity requires OPENML_SKIP_PARQUET=true')
    from openml.datasets.functions import _get_dataset_arff
    path=Path(_get_dataset_arff(dataset))
    expected=dataset.md5_checksum
    require(isinstance(expected,str) and len(expected)==32,'published OpenML MD5 absent')
    md5=hashlib.md5()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):md5.update(b)
    require(md5.hexdigest().lower()==expected.lower(),'OpenML cached ARFF MD5 mismatch '+str(path))
    return {'arff_path':str(path),'published_md5':expected,'verified_md5':md5.hexdigest(),'arff_sha256':sha256(path)}
