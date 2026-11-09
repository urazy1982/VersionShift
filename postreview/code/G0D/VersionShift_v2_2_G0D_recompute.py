#!/usr/bin/env python3
"""Independent read-only recomputation of VersionShift paper point estimates.
Uses only shipped fold-level scores, not manuscript-derived reports.
"""
import argparse, collections, csv, gzip, io, itertools, json, math, os, statistics, zipfile
from pathlib import Path

ALGOS = ('LR','SVC','kNN','RF','GB','HGB')
EDGES = {
 'POS': ('0.21.3','0.22.2.post1'),
 'NEG': ('1.8.0','1.9.1'),
}
PREFIX = 'VersionShift/results/STAGE_C_CPU_COMPLETION/merged/'
EXPECTED = {'POS': {'M':5.01,'affected':59,'winner':25,'flips':165,'material':56},
            'NEG': {'M':0.05,'affected':0,'winner':0,'flips':6,'material':0}}

def to_int(v):
    f=float(v)
    if not f.is_integer(): raise ValueError('Non-integer fold '+str(v))
    return int(f)

def read_rows(z, name):
    with z.open(name) as f, gzip.GzipFile(fileobj=f) as g:
        reader=csv.DictReader(io.TextIOWrapper(g,encoding='utf-8-sig',newline=''))
        for r in reader: yield r

def means_for(z, edge, selected=None, policy='D', algos=ALGOS):
    """Validate exactly 30 unique fold keys per (release,dataset,algorithm)."""
    a,b=EDGES[edge]; scores=collections.defaultdict(dict); errors=[]; status=collections.Counter()
    source=PREFIX+edge+'.csv.gz'
    for r in read_rows(z,source):
        if r.get('policy') != policy or r.get('algo') not in algos: continue
        if r.get('condition') not in (a,b): continue
        if selected is not None and r['dataset'] not in selected: continue
        status[r.get('status','')]+=1
        if r.get('status')!='ok': continue
        k=(r['condition'],r['dataset'],r['algo'])
        try:
            rep,fold=to_int(r['rep']),to_int(r['fold'])
            score=float(r['ba'])
            if not math.isfinite(score) or not (0<=score<=1): raise ValueError('invalid balanced accuracy')
        except (ValueError,TypeError,KeyError) as exc:
            errors.append({'key':str(k),'error':str(exc)}); continue
        fk=(rep,fold)
        if fk in scores[k]:
            if scores[k][fk] != score: errors.append({'key':str(k),'error':'conflicting duplicate fold '+str(fk)})
            else: errors.append({'key':str(k),'error':'duplicate fold '+str(fk)})
        else: scores[k][fk]=score
    incomplete={str(k):len(v) for k,v in scores.items() if len(v)!=30 or set(v)!={(i,j) for i in range(6) for j in range(5)}}
    if incomplete or errors:
        return None, {'source':source,'incomplete':incomplete,'errors':errors[:30], 'status_counts':dict(status)}
    means={k:statistics.mean(v.values())*100 for k,v in scores.items()}
    return means, {'source':source,'cells':len(means),'status_counts':dict(status),'incomplete':{},'errors':[]}

def calculate(means, edge, algos=ALGOS, dataset_set=None):
    a,b=EDGES[edge]
    ids=sorted({d for r,d,al in means if r==a or r==b})
    if dataset_set is not None: ids=sorted(dataset_set)
    pairs=list(itertools.combinations(algos,2))
    results=[]; missing=[]
    for ds in ids:
        if any((r,ds,al) not in means for r in (a,b) for al in algos):
            missing.append(ds);continue
        prev={al:means[a,ds,al] for al in algos}
        post={al:means[b,ds,al] for al in algos}
        shifts=[]; flips=material=0
        for x,y in pairs:
            old=prev[x]-prev[y]; new=post[x]-post[y]
            shifts.append(abs(new-old))
            if (old>0 and new<0) or (old<0 and new>0):
                flips+=1
                if min(abs(old),abs(new))>=1: material+=1
        win_old=max(algos,key=lambda al:(prev[al],-algos.index(al)))
        win_new=max(algos,key=lambda al:(post[al],-algos.index(al)))
        results.append({'dataset':ds,'M_pp':statistics.mean(shifts),
          'winner_changed':win_old!=win_new,'winner_old':win_old,'winner_new':win_new,
          'pair_flips':flips,'material_pair_flips':material})
    out={'edge':edge,'policy':'D','n_algorithms':len(algos),'algorithms':list(algos),
      'K':len(results),'missing_datasets':missing,
      'M_pp':statistics.mean(r['M_pp'] for r in results) if results else None,
      'affected_0_25':sum(r['M_pp']>0.25 for r in results),
      'affected_0_5':sum(r['M_pp']>0.5 for r in results),
      'affected_1_0':sum(r['M_pp']>1 for r in results),
      'winner_changes':sum(r['winner_changed'] for r in results),
      'pair_flips':sum(r['pair_flips'] for r in results),
      'material_pair_flips':sum(r['material_pair_flips'] for r in results),
      'material_flip_tasks':sum(r['material_pair_flips']>0 for r in results),
      'datasets':results}
    return out

def get_holdout(z,edge):
    fn=PREFIX+'HOLDOUT47_'+edge+'.csv.gz'
    return {r['dataset'] for r in read_rows(z,fn)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip',required=True); p.add_argument('--out',required=True)
    args=p.parse_args(); dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    findings={}; fatal=[]
    with zipfile.ZipFile(args.zip) as z:
        for edge in EDGES:
            try:
                means,checks=means_for(z,edge)
                if means is None: fatal.append(edge+' fold integrity failures'); findings[edge]={'fold_checks':checks};continue
                holdout=get_holdout(z,edge)
                whole=calculate(means,edge)
                unseen=calculate(means,edge,dataset_set=holdout)
                pilot=calculate(means,edge,dataset_set=set(d for _,d,_ in means)-holdout)
                cross={}
                if edge in EXPECTED:
                    observed={'M':whole['M_pp'],'affected':whole['affected_0_5'],
                      'winner':whole['winner_changes'],'flips':whole['pair_flips'],
                      'material':whole['material_pair_flips']}
                    for k,v in EXPECTED[edge].items():
                        cross[k]={'expected':v,'observed':observed[k],
                          'match':abs(v-observed[k])<=0.011 if k=='M' else v==observed[k]}
                findings[edge]={'fold_checks':checks,'all63':whole,'holdout47':unseen,
                                'pilot16':pilot,'manuscript_check':cross}
                if whole['missing_datasets'] or whole['K']!=63 or len(holdout)!=47:
                    fatal.append(edge+' dataset coverage incomplete')
            except (KeyError,ValueError,EOFError,zipfile.BadZipFile) as exc:
                fatal.append(edge+': '+repr(exc))
    findings['fatal']=fatal
    (dest/'G0D_metrics.json').write_text(json.dumps(findings,indent=2),encoding='utf-8')
    lines=['VERSION SHIFT G0-D: INDEPENDENT POINT-ESTIMATE RECOMPUTATION',
       'Based on unique status=ok policy=D fold rows; never on published summaries.',
       'No original inputs modified. No confidence intervals recomputed.', '']
    for edge in EDGES:
        r=findings.get(edge,{}); lines.append(edge+':')
        if 'all63' not in r: lines.append('  ERROR '+str(r));continue
        for key in ('all63','pilot16','holdout47'):
            v=r[key]
            lines.append('  %s: K=%s, M=%.5f pp, >0.5=%s, winners=%s, flips=%s, material=%s' %
              (key,v['K'],v['M_pp'],v['affected_0_5'],v['winner_changes'],v['pair_flips'],v['material_pair_flips']))
        lines.append('  Comparison to manuscript: '+str(r['manuscript_check']))
    lines.extend(['','Fatal checks: '+str(fatal),'G0-D complete point-estimate certificate only if no fatals and all manuscript comparisons match.'])
    (dest/'G0D_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    if fatal: raise SystemExit(2)
    if any(not j['match'] for x in EDGES for j in findings[x]['manuscript_check'].values()): raise SystemExit(3)

if __name__=='__main__': main()
