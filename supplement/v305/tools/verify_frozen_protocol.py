#!/usr/bin/env python3
"""Verify exact registered frozen-protocol bytes (offline or online)."""
import argparse,hashlib,pathlib,urllib.request,sys,json
EXPECTED='7a62fb456f89dc58d2797e0e5ce83a456b178b4e740de93e6e9eacfeeca56f55'
URL='https://raw.githubusercontent.com/urazy1982/VersionShift/2330e78171fad7bda3d8bc69a3698204ad35a545/runs/plan/FROZEN_PROTOCOL.json'
def main():
 p=argparse.ArgumentParser(description=__doc__)
 mx=p.add_mutually_exclusive_group(required=True)
 mx.add_argument('--local',type=pathlib.Path,help='original committed file to hash offline')
 mx.add_argument('--fetch',action='store_true',help='download from pinned commit URL (Internet required)')
 a=p.parse_args()
 try:
  raw=a.local.read_bytes() if a.local else urllib.request.urlopen(URL,timeout=30).read()
  actual=hashlib.sha256(raw).hexdigest()
  status='PASS' if actual==EXPECTED else 'FAIL'
  print(json.dumps({'status':status,'url':URL,'expected_sha256':EXPECTED,'actual_sha256':actual,'bytes':len(raw)},indent=2))
  if status!='PASS':sys.exit(1)
 except Exception as e:print('ERROR:',e,file=sys.stderr);sys.exit(2)
if __name__=='__main__':main()
