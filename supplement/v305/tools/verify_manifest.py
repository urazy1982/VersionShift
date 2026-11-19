#!/usr/bin/env python3
"""Verify SHA-256 manifest stored in this companion. Uses no third-party modules."""
import json,pathlib,hashlib,sys,argparse

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);a=p.parse_args()
 root=a.root.resolve();manifest=json.loads((root/'MANIFEST_SHA256.json').read_text())
 got={}
 for rel,expected in manifest['files'].items():
  rp=pathlib.PurePosixPath(rel)
  if rp.is_absolute() or '..' in rp.parts:raise SystemExit('UNSAFE_MANIFEST_PATH '+rel)
  fp=(root/rel).resolve()
  if not fp.is_relative_to(root):raise SystemExit('UNSAFE_MANIFEST_TARGET '+rel)
  if not fp.is_file():raise SystemExit('MISSING '+rel)
  actual=digest(fp)
  if expected!=actual:raise SystemExit('HASH_MISMATCH '+rel)
  got[rel]=actual
 expected_paths=set(got)
 actual_paths={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='MANIFEST_SHA256.json'}
 if expected_paths!=actual_paths:raise SystemExit('FILE_INVENTORY_MISMATCH '+repr(actual_paths^expected_paths))
 print('COMPANION_SHA256_PASS',len(got),'files')
if __name__=='__main__':main()
