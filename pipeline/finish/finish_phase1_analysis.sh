#!/usr/bin/env bash
# VersionShift - finish the Phase 1 analysis (run after `run_versionshift_wsl.sh phase1`).
#
#   bash pipeline/finish/finish_phase1_analysis.sh
#
# Paths default to the layout used in the study; override with VS_PROJECT, VS_DATA, VS_PKG, VS_OUT.
#
# Cause: analysis_v221.validate_plan_coverage() counts only default-policy (D) jobs, so any lens run
# on policy S has 0 "planned default jobs" and divides by zero. This is a defect in the frozen v2.2.5
# analysis code, independent of the data. On 1.8.0 -> 1.9.1 policy S predictions are bitwise identical
# to policy D, so the S lens would equal LENSES_NEG_D_EXPLORATORY.json exactly. This script verifies
# that identity on the real data, records LENSES_NEG_S as NOT_COMPUTED (with the proof), then lets
# the kit finish (emulation fidelity, assessment, output hashes), verifies and bundles.
# The G5 emulation step re-fits 16 datasets in one process and can take a while; to skip it:
#   tr -d '\r' < .../finish_phase1_analysis_v3.sh | VS_SKIP_EMULATION=1 bash
set -Eeuo pipefail
PROJECT="${VS_PROJECT:-/mnt/c/VersionShift_v2_2_5}"
DATA="${VS_DATA:-$HOME/versionshift_data}"
PKG="${VS_PKG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${VS_OUT:-$HOME}"
REP="$DATA/STAGE_C_CPU_COMPLETION/reports"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENML_SKIP_PARQUET=true
source "$HOME/versionshift_controller/bin/activate"
LOG="$DATA/package_logs/finish_analysis_v3_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== finish Phase 1 analysis v3 ($(date -Is)); log=$LOG"
[[ -f "$REP/LENSES_NEG_D_EXPLORATORY.json" ]] || { echo "STOP: LENSES_NEG_D missing - unexpected state; send the log"; exit 2; }

python3 -I - "$REP/LENSES_NEG_S_EXPLORATORY.json" "$DATA/STAGE_C_CPU_COMPLETION/merged/NEG.csv.gz" "$REP/LENSES_NEG_D_EXPLORATORY.json" <<'EOF'
import hashlib, json, sys, time
from pathlib import Path
import pandas as pd
dest, neg, lens_d = map(Path, sys.argv[1:4])
if dest.exists():
    print('kept existing', dest.name); sys.exit(0)
d = pd.read_csv(neg, dtype={'release': str})
k = ['dataset', 'release', 'algo', 'rep', 'fold']
D = d[(d.policy == 'D') & (d.status == 'ok')].set_index(k)
S = d[(d.policy == 'S') & (d.status == 'ok')].set_index(k)
j = D[['pred_sha', 'ba']].join(S[['pred_sha', 'ba']], rsuffix='_S', how='inner')
identical = len(j) == len(D) == len(S) and bool((j.pred_sha == j.pred_sha_S).all()) and float((j.ba - j.ba_S).abs().max()) == 0.0
if not identical:
    sys.exit('STOP: policy S is NOT identical to D on 1.8.0->1.9.1; not writing a record - send this output')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rec = dict(edge='1.8.0:1.9.1', policy='S', status='NOT_COMPUTED_ANALYSIS_DEFECT_EQUIVALENT_TO_D',
           reason='analysis_v221.validate_plan_coverage counts only default-policy (D) jobs; with policy S on both '
                  'releases planned_default_jobs = 0 and the coverage check raises ZeroDivisionError. Defect in frozen '
                  'v2.2.5 analysis code, not a data property.',
           equivalence=dict(fold_rows_compared=int(len(j)), identical_prediction_hashes=True, max_abs_ba_difference=0.0,
                            statement='policy S predictions are bitwise identical to policy D on this edge; the S lens '
                                      'equals LENSES_NEG_D_EXPLORATORY.json'),
           equivalent_report='LENSES_NEG_D_EXPLORATORY.json', equivalent_report_sha256=sha(lens_d),
           input=str(neg), input_sha256=sha(neg), recorded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           role='exploratory lens; no primary inference affected')
dest.write_text(json.dumps(rec, indent=2, sort_keys=True) + '\n')
print('recorded:', dest.name, '-', len(j), 'fold rows identical')
EOF

# re-install the add-on (adds --skip-emulation) and complete the kit analysis
cat > "$PKG/addons/vs_finish_analyze.py" <<'PYEOF'
#!/usr/bin/env python3
"""VersionShift add-on: complete the kit's 'analyze' without editing the frozen kit.

Bug in versionshift_cpu_finish.py combine() (lines 236-240): when a merged CSV already exists, the
kit writes a gzip temp file named '<name>.csv.gz.tmp' and reads it back with pd.read_csv, which
infers compression from the '.tmp' suffix (none) and fails with UnicodeDecodeError 0x8b. It only
triggers when 'analyze' is re-run. This add-on loads the kit module unchanged, replaces combine()
with a version that performs the same checks with explicit gzip compression, removes stale
'*.csv.gz.tmp' leftovers, and then calls the kit's own assess() (which skips existing outputs).
"""
import argparse, importlib.util, sys, tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='/mnt/c/VersionShift_v2_2_5')
    ap.add_argument('--data', default=str(Path.home() / 'versionshift_data'))
    ap.add_argument('--combine-only', action='store_true', help='test: only run the corrected combine check')
    ap.add_argument('--skip-emulation', action='store_true', help='kit option: do not compute G5 (recorded as NOT_PERFORMED)')
    a = ap.parse_args()
    kit_path = Path(a.project) / 'versionshift_cpu_finish.py'
    spec = importlib.util.spec_from_file_location('vs_kit', kit_path)
    K = importlib.util.module_from_spec(spec)
    sys.argv = [str(kit_path)]
    spec.loader.exec_module(K)

    def safe_combine(c):
        import pandas as pd
        merge_dir = c['out'] / 'merged'
        merge_dir.mkdir(parents=True, exist_ok=True)
        for stale in merge_dir.glob('*.csv.gz.tmp'):
            stale.unlink()
            print('removed stale temp file', stale.name)
        result = {}
        for code, old, new, bname in K.EDGES:
            for reverse in (False, True):
                prev = bname + ('_REVERSE' if reverse else '')
                later = 'STAGE_C47_' + code + ('_REVERSE' if reverse else '')
                items = []
                for rn in (prev, later):
                    df = pd.read_csv(c['runs'] / rn / 'CROSS_RELEASE_FOLDS.csv.gz')
                    if set(df['dataset'].astype(str)) != (c['b_ids'] if rn == prev else set(c['remain'])):
                        K.fail('dataset coverage mismatch in ' + rn)
                    items.append(df)
                frame = pd.concat(items, ignore_index=True)
                if set(frame.dataset.astype(str)) != c['all_ids']:
                    K.fail('combined missing datasets')
                if frame[frame.status.eq('ok')].duplicated(['dataset', 'release', 'algo', 'policy', 'rep', 'fold']).any():
                    K.fail('duplicate fold identity in combined frame ' + code)
                if frame.status.eq('ok').sum() == 0:
                    K.fail('no good fold rows')
                dest = merge_dir / (code + ('_REVERSE' if reverse else '') + '.csv.gz')
                if dest.exists():
                    with tempfile.TemporaryDirectory() as td:
                        tmp = Path(td) / 'check.csv.gz'
                        frame.to_csv(tmp, index=False, compression='gzip')
                        same = pd.read_csv(tmp, compression='gzip').equals(pd.read_csv(dest, compression='gzip'))
                    if not same:
                        K.fail('derived CSV changed; preserve existing: ' + str(dest))
                    print('merged file verified unchanged:', dest.name)
                else:
                    frame.to_csv(dest, index=False, compression='gzip')
                    print('merged file written:', dest.name)
                result[code + ('_REVERSE' if reverse else '')] = dest
        return result

    K.combine = safe_combine
    args = argparse.Namespace(command='analyze', project=a.project, data=a.data, workers=10,
                              skip_aa=False, skip_emulation=a.skip_emulation)
    if a.combine_only:
        safe_combine(K.context(args))
        print('COMBINE CHECK OK')
        return
    K.assess(args)
    print('ANALYZE COMPLETE')


if __name__ == '__main__':
    main()
PYEOF
EMU=""; [[ "${VS_SKIP_EMULATION:-}" == "1" ]] && EMU="--skip-emulation" && echo "G5 emulation skipped on request (recorded as NOT_PERFORMED)"
python3 -I "$PKG/addons/vs_finish_analyze.py" --project "$PROJECT" --data "$DATA" $EMU

cd "$PKG"
python3 -I addons/vs_verify_independent.py --data "$DATA" || echo "!! verification reported problems (see VERIFY_SUMMARY.json)"
python3 -I addons/vs_make_upload_bundle.py --data "$DATA" --envs "$HOME/versionshift_envs" --out-dir "$OUT" --label phase1_final
echo "DONE. Results bundle written to $OUT."
