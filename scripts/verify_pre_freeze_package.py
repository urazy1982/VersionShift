#!/usr/bin/env python3
"""Standard-library-only package verifier. Never dereferences host inventories."""
import argparse,csv,hashlib,json
from pathlib import Path,PurePosixPath

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def verify_package(root):
 root=Path(root).resolve();manifest_path=root/'PACKAGE_MANIFEST.json'
 if manifest_path.is_symlink():raise ValueError('manifest symlinks forbidden')
 m=json.loads(manifest_path.read_text())
 if m.get('scope')!='PACKAGE_BYTES_ONLY_NOT_HOST_VERIFICATION':raise ValueError('incorrect package verification scope')
 files=m['files_sha256'];required={'CHANGELOG_16_AUTHOR_DRAFT.csv','E3_PROPOSAL_NOT_ADOPTED.json','verify_package.py','verify_host.py'}
 if not required<=set(files):raise ValueError('required author-review payload absent from manifest')
 if 'PACKAGE_MANIFEST.json' in files:raise ValueError('self-hashing manifest is invalid')
 for name,want in files.items():
  rel=PurePosixPath(name)
  if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:raise ValueError('unsafe package member path')
  path=root.joinpath(*rel.parts)
  if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('package path escapes root')
  if not path.is_file() or sha(path)!=want:raise ValueError('payload missing or SHA differs: '+name)
 actual=set()
 for p in root.rglob('*'):
  if p.is_symlink():raise ValueError('package symlinks forbidden')
  if p.is_file():actual.add(p.relative_to(root).as_posix())
 if actual!=set(files)|{'PACKAGE_MANIFEST.json'}:raise ValueError('unmanifested package files present')
 with (root/'CHANGELOG_16_AUTHOR_DRAFT.csv').open(newline='') as f:rows=list(csv.DictReader(f))
 if len(rows)!=16 or len({(r['edge_id'],r['estimator']) for r in rows})!=16:raise ValueError('not sixteen distinct changelog rows')
 if any(r['author_decision']!='PENDING' for r in rows):raise ValueError('draft author fields unexpectedly decided')
 e3=json.loads((root/'E3_PROPOSAL_NOT_ADOPTED.json').read_text())
 if e3.get('author_confirmed') is not False or e3.get('status')!='PROSPECTIVE_COMPATIBILITY_PROPOSAL_NOT_INSTALLED':raise ValueError('E3 must remain proposal-only')
 return {'status':'PASS_PACKAGE_ONLY','payload_files_verified':len(files),'manifest_sha256':sha(manifest_path),
         'host_files_accessed':False,'host_inventory_verified':False,'scope':m['scope'],
         'qualification':'Integrity relative to the supplied manifest; not a signature, author approval, prospective freeze or host-data replay.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parent));p.add_argument('--out');a=p.parse_args()
 try:r=verify_package(a.root);code=0
 except Exception as ex:r={'status':'FAIL_PACKAGE_ONLY','error':str(ex),'host_inventory_verified':False};code=1
 if a.out:
  path=Path(a.out)
  if path.exists():raise SystemExit('refusing to overwrite verifier output')
  if path.resolve().is_relative_to(Path(a.root).resolve()):raise SystemExit('write verification output outside the package root')
  path.write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r,sort_keys=True));raise SystemExit(code)
if __name__=='__main__':main()
