#!/usr/bin/env python3
"""Separate explicit host-inventory replay. This is NOT package verification."""
import argparse,hashlib,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def verify_host(manifest,workers=8):
 rows=manifest['files'];keys=[r['logical_path'] for r in rows]
 if len(keys)!=len(set(keys)):raise ValueError('duplicate logical host identities')
 def check(r):
  p=Path(r['host_path'])
  try:actual=sha(p);error=None
  except OSError as ex:actual=None;error=str(ex)
  return {'logical_path':r['logical_path'],'expected_sha256':r['sha256'],'actual_sha256':actual,'error':error}
 with ThreadPoolExecutor(max_workers=workers) as pool:checks=list(pool.map(check,rows))
 bad=[r for r in checks if r['expected_sha256']!=r['actual_sha256']]
 return {'status':'PASS_HOST_INVENTORY' if not bad else 'FAIL_HOST_INVENTORY','files_checked':len(rows),'failure_count':len(bad),'failures':bad,'scope':manifest['scope'],'package_verified':False}

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
 if not 1<=a.workers<=8:raise SystemExit('workers must be1..8')
 out=Path(a.out)
 if out.exists():raise SystemExit('refusing to overwrite host verification record')
 r=verify_host(json.loads(Path(a.manifest).read_text()),a.workers);r['manifest_sha256']=sha(Path(a.manifest));out.write_text(json.dumps(r,indent=2)+'\n');print(r['status'],r['files_checked'],'files',r['failure_count'],'failures');raise SystemExit(0 if not r['failures'] else 1)
if __name__=='__main__':main()
