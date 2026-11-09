#!/usr/bin/env python3
"""VersionShift v2.2 G0-C: read-only raw-evidence, hash and job-ledger audit.

Works directly on the original replication ZIP, without extraction or executing
code from the ZIP. Outputs schema/row profiles, verification of the completion
hash manifest, and a job status ledger. This is NOT numerical replication.

  python3 VersionShift_v2_2_G0C_evidence_audit.py \
    --zip /mnt/c/VersionShift/VersionShift_replication.zip \
    --out /mnt/c/VersionShift/V2_2_G0C

Python >=3.8, standard library only. Does not modify the original archive.
"""
from __future__ import annotations
import argparse, collections, csv, gzip, hashlib, io, json, re, sys, zipfile
from pathlib import Path

LOCKED_SHA = '5eb716b64c21ded6f979b0b1610bb68aee46ad2f24889a0d3b133bd430c6ff7d'
C = 'VersionShift/results/STAGE_C_CPU_COMPLETION/'
EXPECT = [
  C+'reports/FULL63_FRAME_CENSUS.json',
  C+'reports/HOLDOUT47_FINITE_SUBFRAME_SENSITIVITY.json',
  C+'reports/CPU_COMPLETION_ASSESSMENT.json',
  C+'CPU_OUTPUT_SHA256.json',
  'VersionShift/results/INDEPENDENT_VERIFICATION/VERIFY_SUMMARY.json',
  'VersionShift/results/data/PLAN.json',
  'VersionShift/results/data/AMENDMENT_V226_FREEZE.json',
  'VersionShift/paper/verify_results.py',
  'VersionShift/paper/reproduce_paper.py',
]

def hashfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(1<<20), b''): h.update(part)
    return h.hexdigest()

def ziphash(z, name):
    h=hashlib.sha256()
    with z.open(name) as f:
        for part in iter(lambda:f.read(1<<20), b''):h.update(part)
    return h.hexdigest()

def loadjson(z,name):
    with z.open(name) as f:return json.load(f)

def sample(s,maxchar=80):
    if s is None:return ''
    s=str(s).replace('\n','\\n').replace('\r','')
    return s[:maxchar]

def traverse(v,path='',max_depth=7):
    if max_depth<0:return
    if isinstance(v,dict):
        for k,val in v.items():
            yield from traverse(val,path+'.'+str(k) if path else str(k),max_depth-1)
    elif isinstance(v,list):
        if len(v)<=8:
            for i,val in enumerate(v):yield from traverse(val,f'{path}[{i}]',max_depth-1)
        else:
            yield path+'.len',len(v)
    elif v is None or isinstance(v,(str,int,float,bool)):
        yield path,v

def member_match(names, manifest_name, relative):
    # The CPU_OUTPUT_SHA256 manifest uses CPU_COMPLETION-relative paths.
    base=manifest_name.rsplit('/',1)[0]+'/'
    proposed=base+relative.lstrip('./')
    return proposed if proposed in names else None

def reported_hash(x):
    if isinstance(x,str) and re.fullmatch('[a-fA-F0-9]{64}',x):return x.lower()
    if isinstance(x,dict):
        for k in ('sha256','sha_256','hash','digest'):
            v=x.get(k)
            if isinstance(v,str) and re.fullmatch('[a-fA-F0-9]{64}',v):return v.lower()
    return None

def status_of(rec):
    if not isinstance(rec,dict):return 'NON_OBJECT'
    for k in ('status','state','outcome','job_status'):
        if k in rec:
            val=rec[k]
            if isinstance(val,dict):return sample(val,75)
            return sample(val,75)
    return 'UNSPECIFIED'

def csv_profile(z,name):
    gz=name.lower().endswith('.gz')
    counts=collections.Counter(); nulls=collections.Counter(); samples={}
    read_count=0
    with z.open(name) as raw:
        stream=gzip.GzipFile(fileobj=raw,mode='rb') if gz else raw
        textstream=io.TextIOWrapper(stream,encoding='utf-8-sig',errors='replace',newline='')
        reader=csv.DictReader(textstream)
        fields=reader.fieldnames or []
        if not fields: raise ValueError('CSV missing header')
        for rec in reader:
            read_count+=1
            for k,v in rec.items():
                if k is None:continue
                if v is None or v=='':nulls[k]+=1
                if k not in samples:samples[k]=[]
                if v is not None and len(samples[k])<3 and sample(v) not in samples[k]:samples[k].append(sample(v))
            if not isinstance(rec,dict):raise ValueError('Unexpected CSV row')
    return {'path':name,'kind':('csv.gz' if gz else 'csv'), 'rows':read_count,'columns':fields,'null_cells':dict(nulls),'first_values':samples}

def jsonl_profile(z,name):
    counts=collections.Counter(); rows=0; invalid=[]; keys=collections.Counter(); examples=[]
    with z.open(name) as fp:
        for i,line in enumerate(fp,1):
            if not line.strip():continue
            rows+=1
            try:rec=json.loads(line)
            except Exception as e:
                if len(invalid)<10:invalid.append({'line':i,'error':type(e).__name__})
                continue
            counts[status_of(rec)]+=1
            if isinstance(rec,dict):keys.update(rec.keys())
            if len(examples)<2:examples.append(rec)
    return {'path':name,'rows':rows,'status_counts':dict(counts),'invalid_lines':invalid,
            'keys':dict(keys),'samples':examples}

def csv_save(path,header,rows):
    with path.open('w',newline='',encoding='utf-8') as fp:
        w=csv.writer(fp);w.writerow(header);w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip',required=True,dest='zip_path')
    ap.add_argument('--out',required=True)
    ap.add_argument('--expected-sha256',default=LOCKED_SHA,help='Locked original archive SHA256; empty to bypass only for synthetic tests')
    args=ap.parse_args(); source=Path(args.zip_path).expanduser();out=Path(args.out).expanduser()
    if not source.is_file():sys.exit(f'STOP: archive not found: {source}')
    digest=hashfile(source)
    if args.expected_sha256 and digest != args.expected_sha256.lower():sys.exit(f'STOP: archive changed! SHA-256 {digest}')
    out.mkdir(parents=True,exist_ok=True)
    payload={'tool':'VersionShift v2.2 G0C evidence audit','zip':str(source),'sha256':digest,'read_only':True,
             'disclaimer':'Structural inspection and hash verification only. NOT a numerical/scientific replication.'}
    warnings=[]; profiles=[]; statuses=[]; reports={}; sha_records=[]
    with zipfile.ZipFile(source) as z:
        names=set(i.filename for i in z.infolist() if not i.is_dir())
        missing=[x for x in EXPECT if x not in names]
        payload['required_files']={'present':len(EXPECT)-len(missing),'expected':len(EXPECT),'missing':missing}
        for member in sorted(names):
            if member.endswith('.csv.gz') or (member.endswith('.csv') and '/merged/' in member):
                try: profiles.append(csv_profile(z,member))
                except Exception as exc:
                    profiles.append({'path':member,'error':type(exc).__name__+': '+str(exc)[:240]})
                    warnings.append('Could not profile CSV: '+member)
            if member.endswith('.jsonl') and ('STAGE_B_RUNS/' in member or 'JOB_STATUS' in member):
                try:statuses.append(jsonl_profile(z,member))
                except Exception as exc:
                    statuses.append({'path':member,'error':type(exc).__name__+': '+str(exc)[:240]})
                    warnings.append('Could not profile JSONL: '+member)
        target=C+'CPU_OUTPUT_SHA256.json'
        if target in names:
            try:
                manifest=loadjson(z,target)
                if not isinstance(manifest,dict):raise ValueError('Expected hash mapping object')
                for relative, recorded in manifest.items():
                    expected=reported_hash(recorded)
                    path=member_match(names,target,relative)
                    if expected is None:status='HASH_NOT_RECOGNIZED'; actual=''
                    elif path is None:status='MEMBER_MISSING';actual=''
                    else:
                        actual=ziphash(z,path)
                        status='PASS' if actual==expected else 'MISMATCH'
                    sha_records.append([relative,path or '',expected or '',actual,status])
            except Exception as e:warnings.append('Could not check CPU_OUTPUT_SHA256: '+str(e)[:240])
        else:warnings.append('Missing CPU_OUTPUT_SHA256 manifest')
        desired=[C+'reports/FULL63_FRAME_CENSUS.json',C+'reports/HOLDOUT47_FINITE_SUBFRAME_SENSITIVITY.json',
                 C+'reports/CPU_COMPLETION_ASSESSMENT.json',
                 'VersionShift/results/INDEPENDENT_VERIFICATION/VERIFY_SUMMARY.json',
                 'VersionShift/results/data/STAGE_C_COMPLETION_PROTOCOL_RECORD.json',
                 'VersionShift/results/data/AMENDMENT_V226_FREEZE.json']
        for name in desired:
            if name in names:
                try:reports[name]=loadjson(z,name)
                except Exception as exc:warnings.append('Cannot decode JSON '+name+': '+str(exc)[:180])
        collect=[]
        for name in sorted(names):
            if name.endswith('COLLECT_SUMMARY.json') and '/STAGE_B_RUNS/' in name:
                try:
                    d=loadjson(z,name)
                    collect.append({'path':name,'planned_jobs':d.get('planned_jobs'),
                       'missing_jobs':d.get('missing_jobs'), 'status_counts':d.get('status_counts',{}),
                       'portability_counts':d.get('portability_counts',{})})
                except Exception as exc:warnings.append('Cannot decode collect summary '+name+': '+str(exc)[:180])
    csv_save(out/'G0C_hash_checks.csv',['source_relative','zip_member','expected_sha256','actual_sha256','status'],sha_records)
    csv_save(out/'G0C_compressed_rows.csv',['path','rows','columns','error'],[
        [p['path'],p.get('rows',''),'|'.join(p.get('columns',[])),p.get('error','')] for p in profiles])
    csv_save(out/'G0C_job_ledger.csv',['run','planned_jobs','missing_jobs','status_counts','portability_counts'],[
        [p['path'],p['planned_jobs'],p['missing_jobs'],json.dumps(p['status_counts']),json.dumps(p['portability_counts'])] for p in collect])
    scalars=[]
    for name,report in reports.items():
        for key,v in traverse(report):
            if any(k in key.lower() for k in ('mean','score','shift','flip','winner','rank','m_pp','holdout','fold','dataset','task','coverage','status','frozen','success','change','job','gate','problem','error','ident','total','pass','fail')):
                scalars.append([name,key,sample(v,260)])
    csv_save(out/'G0C_report_fields.csv',['report','json_path','value'],scalars)
    payload['compressed_file_profiles']=profiles
    payload['job_status_profiles']=statuses
    payload['collect_summaries']=collect
    payload['completion_hash_checks']={'entries':len(sha_records),'pass':sum(r[-1]=='PASS' for r in sha_records),
       'fail':sum(r[-1] not in ('PASS',) for r in sha_records)}
    payload['reports_loaded']=list(reports)
    payload['warnings']=warnings
    (out/'G0C_profiles.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
    lines=['VersionShift v2.2 G0-C RAW-EVIDENCE AUDIT',
       'Original ZIP: '+str(source),'SHA256: '+digest,
       'GATE: G0-C is an evidence audit, NOT independent numerical replication.','',
       'REQUIRED FILES: '+str(payload['required_files']['present'])+'/'+str(len(EXPECT)),
       'MISSING: '+(', '.join(missing) if missing else 'none'),
       f'CPU OUTPUT HASHES: {payload["completion_hash_checks"]["pass"]}/{payload["completion_hash_checks"]["entries"]} PASS',
       'WARNINGS: '+str(len(warnings)),
       '', 'COMPRESSED CSV ROWS — EACH FILE IS A SEPARATE SOURCE (DO NOT SUM OVERLAPPING VERSIONS):']
    for p in profiles:
        lines.append(f'  {p["path"]}: rows={p.get("rows","ERROR")} cols={"|".join(p.get("columns",[]))[:250]}'+(' ERROR '+p['error'] if 'error' in p else ''))
    lines+=['','RUN COLLECTION COUNTS (may contain duplicated AA attempts; DO NOT sum blindly):']
    for c in collect:
        lines.append('  '+c['path'].split('/STAGE_B_RUNS/')[-1].split('/')[0]+f': planned={c["planned_jobs"]} missing={c["missing_jobs"]} statuses={c["status_counts"]} portable={c["portability_counts"]}')
    lines+=['','REPORTS LOADED:']+['  '+n for n in reports]
    lines+=['','SCIENTIFIC CERTIFICATION: NOT YET PERFORMED',
            'NEXT: Use G0C_profiles.json + G0C_report_fields.csv to write a schema-specific independent recomputation (G0-D).',
            'Original ZIP and original results were not modified.']
    (out/'G0C_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:12]))
    if payload['completion_hash_checks']['fail'] or missing or warnings:
        print('ATTENTION: Audit contains missing/failed items. Read G0C_summary.txt and G0C_hash_checks.csv.')
    print('Outputs:',out)

if __name__=='__main__':main()
