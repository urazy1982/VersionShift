import hashlib
from types import SimpleNamespace
import pytest
from versionshift_v3.source_integrity import verify_arff

def test_cached_arff_integrity(tmp_path,monkeypatch):
    import openml.datasets.functions as functions
    p=tmp_path/'dataset.arff';p.write_bytes(b'@relation test\n@data\n1\n')
    monkeypatch.setenv('OPENML_SKIP_PARQUET','true')
    monkeypatch.setattr(functions,'_get_dataset_arff',lambda ds:p)
    ds=SimpleNamespace(md5_checksum=hashlib.md5(p.read_bytes()).hexdigest())
    assert verify_arff(ds)['arff_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    p.write_bytes(b'corrupt cache')
    with pytest.raises(RuntimeError,match='MD5 mismatch'):verify_arff(ds)

def test_missing_checksum_is_not_silently_accepted(tmp_path,monkeypatch):
    import openml.datasets.functions as functions
    p=tmp_path/'dataset.arff';p.write_bytes(b'data')
    monkeypatch.setenv('OPENML_SKIP_PARQUET','true')
    monkeypatch.setattr(functions,'_get_dataset_arff',lambda ds:p)
    with pytest.raises(RuntimeError,match='MD5 absent'):verify_arff(SimpleNamespace(md5_checksum=None))
