#!/usr/bin/env python3
"""Read-only scientific-evidence inventory of a VersionShift replication ZIP.

This script profiles ZIP members *in memory*, never extracts paths, never executes
source files, and never modifies the ZIP. It is NOT a scientific certification.

Usage:
 python3 VersionShift_v2_2_G0B_inventory.py \
    --zip /mnt/c/VersionShift/VersionShift_replication.zip \
    --out /mnt/c/VersionShift/V2_2_G0B
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

EXPECTED_PUBLICATION = {
    "tasks": 63, "folds_per_task": 30, "scheduled_jobs": 8520,
    "reported_fold_records": 226368, "Eplus_M_pp": 5.01,
    "Ezero_M_pp": 0.05, "Eplus_winner_changes": 25,
    "Eplus_order_flips": 165, "Eplus_material_flips": 56,
}
TEXT_EXTS = {'.py','.sh','.bash','.yaml','.yml','.toml','.ini','.cfg','.conf',
             '.json','.jsonl','.ndjson','.csv','.tsv','.txt','.md','.rst','.tex',
             '.bib','.lock','.log','.sha256','.gitignore','.xml'}
KEYWORDS = {
 "raw_results": r'fold|prediction|scores?|raw[_-]?result|results?|records?|output|run[_-]?data',
 "scripts": r'\.py$|\.sh$|analysis|verify|audit|recomput|runner|pipeline|repro',
 "frozen_protocol": r'protocol|freeze|amend|prereg|hash|sha256|manifest|plan|split|seed',
 "environments": r'environ|lock|conda|requirements|provenance|docker|container',
 "paper": r'paper|manuscript|fig|table|\.tex$|\.bib$|\.pdf$',
 "summary": r'summary|report|aggregate|metric|primary|conclusion|attribut|policy|mitigat',
}
IGNORE_FOR_OVERVIEW = ('.git/', '__pycache__/', '.DS_Store')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def sample_shape(value, depth=0):
    if depth > 2:
        return type(value).__name__
    if isinstance(value, dict):
        out={'type':'object','num_keys':len(value),'keys':list(value)[:60]}
        # Representative values of the first 12 keys only
        out['sample_fields']={str(k):sample_shape(value[k],depth+1) for k in list(value)[:12]}
        return out
    if isinstance(value, list):
        return {'type':'array','length':len(value),'first':sample_shape(value[0],depth+1) if value else None}
    if isinstance(value,str):
        return {'type':'string','sample':value[:120]}
    return {'type':type(value).__name__, 'sample':value}

def profile_member(z, info):
    name=info.filename
    suffix=Path(name).suffix.lower()
    base={'name':name,'bytes':info.file_size,'suffix':suffix,
          'categories':[k for k,p in KEYWORDS.items() if re.search(p,name,re.I)]}
    if suffix not in TEXT_EXTS or info.file_size > 12_000_000:
        base['profile']='binary_or_unsupported' if suffix not in TEXT_EXTS else 'over_size_limit'
        return base
    try:
        with z.open(info) as fp:
            blob=fp.read(12_000_001)
        if len(blob)>12_000_000:
            base['profile']='over_size_limit'
            return base
        s=blob.decode('utf-8-sig',errors='replace')
        lines=s.count('\n')+(1 if s and not s.endswith('\n') else 0)
        base.update(text_lines=lines,profile='text')
        if suffix == '.json':
            try:
                d=json.loads(s)
                base['profile']='json'
                base['structure']=sample_shape(d)
            except Exception as exc:
                base['parse_warning']=f'{type(exc).__name__}: {str(exc)[:150]}'
        elif suffix in ('.csv','.tsv'):
            try:
                dialect='excel-tab' if suffix=='.tsv' else 'excel'
                reader=csv.reader(io.StringIO(s), dialect=dialect)
                first=next(reader,[])
                count=sum(1 for _ in reader)
                base['profile']='csv'
                base['columns']=first[:100]
                base['data_rows']=count
            except Exception as exc:
                base['parse_warning']=f'{type(exc).__name__}: {str(exc)[:150]}'
        elif suffix in ('.jsonl','.ndjson'):
            seen=[]; bad=[]; count=0
            for line in s.splitlines():
                if not line.strip():continue
                count+=1
                try:
                    if len(seen)<2: seen.append(sample_shape(json.loads(line)))
                    else: json.loads(line)
                except Exception:
                    if len(bad)<5:bad.append(count)
            base.update(profile='jsonl',data_rows=count,examples=seen,invalid_line_numbers=bad)
        elif suffix=='.py':
            base['entrypoint']=bool(re.search(r'if\s+__name__\s*==\s*[\'\"]__main__',s))
            base['imports']=sorted(set(re.findall(r'^(?:from|import)\s+([\w.]+)',s,re.M)))[:25]
            base['functions']=re.findall(r'^def\s+([\w]+)\s*\(',s,re.M)[:25]
            base['classes']=re.findall(r'^class\s+([\w]+)',s,re.M)[:25]
        elif suffix in ('.yaml','.yml'):
            base['top_keys']=re.findall(r'^([A-Za-z_][\w\-]*)\s*:',s,re.M)[:40]
        else:
            base['first_lines']=[v[:220] for v in s.splitlines()[:3]]
    except Exception as exc:
        base['profile']='read_failure'
        base['parse_warning']=f'{type(exc).__name__}: {str(exc)[:180]}'
    return base

def rank(p):
    cat=p['categories']; name=p['name'].lower()
    score=0
    if 'raw_results' in cat: score+=5
    if 'summary' in cat: score+=4
    if 'frozen_protocol' in cat: score+=3
    if 'scripts' in cat:score+=3
    if 'environments' in cat:score+=2
    if p.get('profile') in ('json','csv','jsonl'):score+=2
    if name.endswith('.pdf'):score-=10
    if 'test' in name:score-=1
    if 'readme' in name:score+=3
    return -score,name

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',required=True,dest='source')
    parser.add_argument('--out',required=True)
    parser.add_argument('--expected-sha256',default='5eb716b64c21ded6f979b0b1610bb68aee46ad2f24889a0d3b133bd430c6ff7d')
    a=parser.parse_args(); source=Path(a.source).expanduser(); out=Path(a.out).expanduser()
    if not source.is_file():sys.exit('ZIP not found: '+str(source))
    got=sha256(source)
    if a.expected_sha256 and got.lower()!=a.expected_sha256.lower():
        sys.exit('STOP: archive SHA-256 differs from approved G0-A package: '+got)
    out.mkdir(parents=True,exist_ok=True)
    profiles=[]; path_count=collections.Counter(); cats=collections.Counter(); errors=[]
    with zipfile.ZipFile(source,'r') as z:
        infos=[i for i in z.infolist() if not i.is_dir()]
        for info in infos:
            p=profile_member(z,info)
            profiles.append(p)
            top=p['name'].strip('/').split('/')[0]
            path_count[top]+=1
            for c in p['categories']:cats[c]+=1
            if p['profile']=='read_failure':errors.append(p['name'])
    profiles.sort(key=lambda p:p['name'])
    data={'schema_version':1,'source_path':str(source),'sha256':got,
          'count':len(profiles),'uncompressed_bytes':sum(p['bytes'] for p in profiles),
          'top_level_members':dict(path_count.most_common()),
          'category_counts':dict(cats.most_common()),
          'reported_manuscript_targets_NOT_YET_VERIFIED':EXPECTED_PUBLICATION,
          'file_profiles':profiles,
          'disclaimer':'This inventories source and data structures; no scientific metrics are reproduced.'}
    (out/'G0B_profiles.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
    with (out/'G0B_file_index.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['path','size_bytes','category','profile','rows_or_lines','top_level_keys_or_columns'])
        for p in profiles:
            keys=p.get('columns') or p.get('top_keys') or p.get('structure',{}).get('keys',[])
            w.writerow([p['name'],p['bytes'],';'.join(p['categories']),p['profile'],p.get('data_rows',p.get('text_lines','')),';'.join(map(str,keys))])
    candidates=sorted(profiles,key=rank)
    lines=[
        'VersionShift v2.2 G0-B SCIENTIFIC PACKAGE INVENTORY',
        'ZIP SHA-256: '+got,
        f'Files: {len(profiles)}; uncompressed bytes: {data["uncompressed_bytes"]:,}',
        'This is file/schema discovery, NOT analysis replication or scientific certification.',
        '', 'TOP-LEVEL MEMBER COUNTS:',
    ]
    lines += [f'  {k}: {v}' for k,v in path_count.most_common()]
    lines += ['', 'CATEGORY COUNTS (overlapping):']
    lines += [f'  {k}: {v}' for k,v in cats.most_common()]
    lines += ['', 'TOP 80 EVIDENCE CANDIDATES (path | bytes | type | records/lines | sample keys):']
    for p in candidates[:80]:
        keys=p.get('columns') or p.get('top_keys') or p.get('structure',{}).get('keys',[])
        lines.append('  '+p['name']+' | '+str(p['bytes'])+' | '+p['profile']+' | '+str(p.get('data_rows',p.get('text_lines','?')))+' | '+', '.join(map(str,keys[:25])))
    lines+=['','POTENTIAL SCRIPT ENTRY POINTS:']
    lines.extend('  '+p['name'] for p in profiles if p.get('entrypoint'))
    lines+=['','FILES WITH STRUCTURED CONTENT:']
    for p in profiles:
        if p['profile'] in ('json','jsonl','csv'):
            fmt = p.get('structure',{}).get('type','')
            if fmt=='array':fmt+=f'[{p["structure"]["length"]}]'
            if fmt=='object':fmt+=f' ({p["structure"]["num_keys"]} keys)'
            lines.append('  '+p['name']+' | '+p['profile']+' | '+str(p.get('data_rows',''))+' | '+fmt)
    lines+=['','NEXT GATE: identify authoritative fold rows and job statuses, then recompute v2.1 paper metrics.',
            'Original archive must remain unchanged; do not overwrite original results.']
    (out/'G0B_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:min(len(lines),28)]))
    print('\nSaved: '+str(out/'G0B_summary.txt'))
    print('Saved: '+str(out/'G0B_profiles.json'))
    print('Saved: '+str(out/'G0B_file_index.csv'))
    if errors:sys.exit('READ ERRORS: '+', '.join(errors[:20]))
if __name__=='__main__': main()
