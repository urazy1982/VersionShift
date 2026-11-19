#!/usr/bin/env python3
"""Generate a REVIEW-REQUIRED JSON view of the v2.6 G3E EXPLICIT map.
The original .py source remains the byte-authoritative E2 definition.
"""
import ast,argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from versionshift_v3.utils import write_once,sha256

def literal(node):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and not node.args:
        return {k.arg:literal(k.value) for k in node.keywords}
    if isinstance(node,ast.Dict):return {literal(k):literal(v) for k,v in zip(node.keys,node.values)}
    if isinstance(node,(ast.Tuple,ast.List)):return [literal(x) for x in node.elts]
    return ast.literal_eval(node)

def extract(source):
    a=ast.parse(source.read_text())
    assignments=[n.value for n in a.body if isinstance(n,ast.Assign) and
                 any(isinstance(t,ast.Name) and t.id=='EXPLICIT' for t in n.targets)]
    if len(assignments)!=1:raise RuntimeError('expected exactly one EXPLICIT assignment')
    mp=literal(assignments[0]);expected={'LR','SVC','kNN','RF','GB','HGB'}
    if set(mp)!=expected:raise RuntimeError('E2 six-estimator key mismatch: '+repr(set(mp)))
    return {'provenance':'DERIVED_FROM_v26_G3E_WORKER_NOT_BYTE_IDENTICAL_TO_ORIGINAL_JSON',
            'source_path':str(source.resolve()),'source_sha256':sha256(source),
            'parameters':mp,'author_confirmed':False,'outcome_informed_v26':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    r=extract(Path(a.worker));write_once(a.out,r)
    print('EXTRACTED_REVIEW_REQUIRED',len(r['parameters']),'algorithms; source_sha256',r['source_sha256'])
if __name__=='__main__':main()
