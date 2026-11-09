#!/usr/bin/env bash
# VersionShift v2.2.5 read-only Stage B scientific analysis.
# This script reads collected data and writes new reports. It does not alter the four run dirs.
set -Eeuo pipefail

PROJECT="${VS_PROJECT:-/mnt/c/VersionShift_v2_2_5}"
DATA="${VS_DATA:-$HOME/versionshift_data}"
PLAN="$DATA/STAGE_B_PLAN_CACHED"
SEL="$DATA/STAGE_B_SELECTION.json"
RUNS="$DATA/STAGE_B_RUNS"
ORACLE="$PROJECT/ORACLE_FROZEN.json"
OUT="$DATA/STAGE_B_SCIENCE_V225"
ANALYSIS="$PROJECT/code/analysis_v221.py"
PY="${VS_CONTROLLER_PYTHON:-$HOME/versionshift_controller/bin/python}"
FORWARD_POS="$RUNS/EDGE_0213_0222/CROSS_RELEASE_FOLDS.csv.gz"
REVERSE_POS="$RUNS/EDGE_0213_0222_REVERSE/CROSS_RELEASE_FOLDS.csv.gz"
FORWARD_NEG="$RUNS/EDGE_180_190/CROSS_RELEASE_FOLDS.csv.gz"
REVERSE_NEG="$RUNS/EDGE_180_190_REVERSE/CROSS_RELEASE_FOLDS.csv.gz"

for item in "$PY" "$ANALYSIS" "$PLAN/PLAN.json" "$SEL" "$ORACLE" \
    "$PROJECT/ORACLE_FRAME_REVIEWED.csv" "$PROJECT/versionshift_policy_descriptives.py" "$FORWARD_POS" "$REVERSE_POS" "$FORWARD_NEG" "$REVERSE_NEG"; do
    [[ -f "$item" ]] || { echo "STOP missing: $item" >&2; exit 2; }
done

if [[ -e "$OUT/EDGES_HOLM.json" ]]; then
    echo "STOP: existing scientific analysis reports would be overwritten in $OUT" >&2
    exit 2
fi
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Verify local oracle freeze and plan/selection binding before reading outcomes.
"$PY" - "$PLAN/PLAN.json" "$SEL" "$ORACLE" "$RUNS" <<'PY'
import hashlib,json,sys,pathlib
plan,selection,frozen,runs=map(pathlib.Path,sys.argv[1:])
sh=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads(plan.read_text()); s=json.loads(selection.read_text()); f=json.loads(frozen.read_text())
assert sh(plan)==s['plan_sha256'], 'selection does not match frozen plan'
assert len(s['stage_b_datasets'])==len(set(s['stage_b_datasets']))==16, 'selection is not 16 unique datasets'
frame=pathlib.Path(f['frame_path'])
assert f['status']=='FROZEN_COMPLETE' and sh(frame)==f['frame_sha256'], 'frozen oracle has changed or is missing'
for name,policies,expected_pin in [
    ('EDGE_0213_0222',{'D','E','P','S'},'0.21.3'),
    ('EDGE_0213_0222_REVERSE',{'P'},'0.22.2.post1'),
    ('EDGE_180_190',{'D','E','P','S'},'1.8.0'),
    ('EDGE_180_190_REVERSE',{'P'},'1.9.1')]:
    root=runs/name
    lock=json.loads((root/'RUN_LOCK.json').read_text())
    summary=json.loads((root/'COLLECT_SUMMARY.json').read_text())
    assert lock['plan_sha256']==sh(plan), name+' plan lock mismatch'
    assert lock['selection_sha256']==sh(selection), name+' selection lock mismatch'
    assert set(lock['selected_datasets'])==set(s['stage_b_datasets']), name+' selected IDs mismatch'
    assert set(lock['policies'])==policies, name+' unexpected policy set'
    assert lock['pin_from']==expected_pin, name+' pin direction mismatch'
    assert summary['planned_jobs']==16*len(lock['releases'])*len(lock['algorithms'])*len(lock['policies']), name+' job-count mismatch'
    assert not summary.get('missing_jobs'), name+' missing job records'
    assert sum(summary['status_counts'].values())==summary['planned_jobs'], name+' unaccounted jobs'
print('PRECHECK OK: plan',sh(plan),'selection',sh(selection),'oracle',f['frame_sha256'])
print('PRECHECK OK: four locked and collected runs, 1920/1920 job records')
PY

# Primary edge analysis, jointly Holm-adjusted across two preregistered edges.
"$PY" "$ANALYSIS" edge \
   --inp "$FORWARD_POS" "$FORWARD_NEG" \
   --edges '0.21.3:0.22.2.post1,1.8.0:1.9.1' \
   --primary '0.21.3:0.22.2.post1,1.8.0:1.9.1' \
   --population stratified --selection "$SEL" --plan-dir "$PLAN" \
   --oracle-frozen "$ORACLE" --out "$OUT/EDGES_HOLM.json"

# Mitigation in strictly portable P population. The PT policy was NOT executed.
"$PY" "$ANALYSIS" mitigation \
   --inp "$FORWARD_POS" --edge '0.21.3:0.22.2.post1' \
   --no-pt --plan-dir "$PLAN" --selection "$SEL" \
   --oracle-frozen "$ORACLE" --out "$OUT/MITIGATION_0213_0222.json"
"$PY" "$ANALYSIS" mitigation \
   --inp "$FORWARD_NEG" --edge '1.8.0:1.9.1' \
   --no-pt --plan-dir "$PLAN" --selection "$SEL" \
   --oracle-frozen "$ORACLE" --out "$OUT/MITIGATION_180_190.json"

# Secondary descriptive D/E/P/S comparisons (not pooled with the P-portability estimand).
"$PY" "$PROJECT/versionshift_policy_descriptives.py" --project "$PROJECT" \
    --positive "$FORWARD_POS" --later "$FORWARD_NEG" \
    --out "$OUT/POLICIES_DESCRIPTIVE.json"

# The frozen attribution validator requires corresponding D records whenever a P cell is read.
# Reverse-P-only runs lack D. Build separately labelled, temporary C-cell inputs by attaching
# old-release D status/folds from the forward run. No scores, statuses or predictions are changed.
"$PY" - "$FORWARD_POS" "$REVERSE_POS" "$OUT/CELL_C_POS.csv.gz" \
           "$FORWARD_NEG" "$REVERSE_NEG" "$OUT/CELL_C_NEG.csv.gz" <<'PY'
import sys,pandas as pd
for forward,reverse,target in ((sys.argv[1],sys.argv[2],sys.argv[3]),
                               (sys.argv[4],sys.argv[5],sys.argv[6])):
    f=pd.read_csv(forward); r=pd.read_csv(reverse)
    old=r['release'].dropna().unique()
    if len(old)!=2: raise SystemExit('STOP: reverse run lacks exactly two releases')
    forward_d=f[f.policy.eq('D')]
    reverse_p=r[r.policy.eq('P')]
    # In the reverse run P is pinned from the NEW release; use its OLD-runtime P records only.
    # New release is identified by sorted-release order as it appears in forward lock table.
    order={"0.21.3":"0.21.3", "0.22.2.post1":"0.21.3", "1.8.0":"1.8.0", "1.9.1":"1.8.0"}
    if not set(old)<=set(order): raise SystemExit('STOP: unexpected release values')
    old_release=order[old[0]]
    old_D=forward_d[forward_d.release.eq(old_release)]
    old_P=reverse_p[reverse_p.release.eq(old_release)]
    if old_D.empty or old_P.empty: raise SystemExit('STOP: old D or reverse old P missing')
    key=['dataset','release','policy','algo','rep','fold']
    full=pd.concat([old_D,old_P],ignore_index=True)
    if full[full.status.eq('ok')].duplicated(key).any(): raise SystemExit('STOP: duplicate cell-C record')
    full.to_csv(target,index=False,compression='gzip')
    print('C-cell input:',target,'D records',len(old_D),'reverse P records',len(old_P))
PY

"$PY" "$ANALYSIS" attribution \
   --cell "A=$FORWARD_POS@0.21.3:D" \
   --cell "B=$FORWARD_POS@0.22.2.post1:P" \
   --cell "C=$OUT/CELL_C_POS.csv.gz@0.21.3:P" \
   --cell "D=$FORWARD_POS@0.22.2.post1:D" \
   --plan-dir "$PLAN" --selection "$SEL" --oracle-frozen "$ORACLE" \
   --out "$OUT/ATTRIBUTION_0213_0222.json"

"$PY" "$ANALYSIS" attribution \
   --cell "A=$FORWARD_NEG@1.8.0:D" \
   --cell "B=$FORWARD_NEG@1.9.1:P" \
   --cell "C=$OUT/CELL_C_NEG.csv.gz@1.8.0:P" \
   --cell "D=$FORWARD_NEG@1.9.1:D" \
   --plan-dir "$PLAN" --selection "$SEL" --oracle-frozen "$ORACLE" \
   --out "$OUT/ATTRIBUTION_180_190.json"

# Size uses pre-established primary-family edge labels, not re-running independent single-edge tests.
"$PY" - "$OUT/EDGES_HOLM.json" "$OUT/EDGE_POS_FOR_SIZE.json" "$OUT/EDGE_NEG_FOR_SIZE.json" <<'PY'
import json,sys,pathlib
p=pathlib.Path(sys.argv[1]); full=json.loads(p.read_text())
for edge,out in [('0.21.3:0.22.2.post1',sys.argv[2]),('1.8.0:1.9.1',sys.argv[3])]:
    q={'margins': {'0.5': {edge:full['margins']['0.5'][edge]}}}
    pathlib.Path(out).write_text(json.dumps(q,indent=2))
PY
"$PY" "$ANALYSIS" size \
   --negative "$OUT/EDGE_NEG_FOR_SIZE.json" \
   --positive "$OUT/EDGE_POS_FOR_SIZE.json" \
   --eligible 63 --stage-b-used 16 \
   --out "$OUT/STAGE_C_SIZE_EXPLORATORY.json"

# Summarize without changing any scientific verdict.
"$PY" - "$OUT" <<'PY'
from pathlib import Path
import json,sys
p=Path(sys.argv[1])
for name in ['EDGES_HOLM.json','MITIGATION_0213_0222.json','MITIGATION_180_190.json',
             'ATTRIBUTION_0213_0222.json','ATTRIBUTION_180_190.json','POLICIES_DESCRIPTIVE.json','STAGE_C_SIZE_EXPLORATORY.json']:
    j=json.loads((p/name).read_text())
    print('\n===',name,'===')
    if name=='EDGES_HOLM.json':
        for edge,r in j['margins']['0.5'].items():
            print(edge,'M_pp=',r['q_mean']['M_pp'],'label=',r.get('label'),'K=',r['K_complete_case'])
    elif name.startswith('MITIGATION'):
        print('coverage=',j.get('plan_coverage',{}).get('status'),'classes=',j.get('failure_to_transfer',{}).get('totals'))
        print('strict P=',j.get('mitigation_P'))
    elif name.startswith('ATTRIBUTION'):
        print('identifiability=',j['identifiability_counts'])
    elif name=='POLICIES_DESCRIPTIVE.json': print(j)
    else: print('K_required=',j.get('K_required'),'decision=',j.get('decision'))
print('\nReports:',p)
PY

(
  cd "$OUT"
  sha256sum EDGES_HOLM.json MITIGATION_*.json ATTRIBUTION_*.json POLICIES_DESCRIPTIVE.json STAGE_C_SIZE_EXPLORATORY.json \
    > REPORT_SHA256.txt
)
echo "STAGE B SCIENTIFIC ANALYSIS COMPLETE: $OUT"
