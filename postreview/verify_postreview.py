#!/usr/bin/env python3
"""Re-run every read-only post-review analysis and check it reproduces the archived outputs in raw/.

    python3 postreview/verify_postreview.py --zip VersionShift_replication.zip [--plan-dir ~/versionshift_data/STAGE_B_PLAN_CACHED]

--zip       the frozen original archive (SHA-256 5eb716b6...), attached to the first release
--plan-dir  optional: the rebuilt plan cache (datasets + folds). Enables the G3-G replay of all 126 E2 HGB jobs,
            which recomputes every fold's balanced accuracy from the stored predictions.

Nothing in raw/ is modified; outputs are written to a temporary directory. A few seconds without --plan-dir.
"""
import argparse, csv, filecmp, hashlib, json, math, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE, RAW = HERE / 'code', HERE / 'raw'
FROZEN = '5eb716b64c21ded6f979b0b1610bb68aee46ad2f24889a0d3b133bd430c6ff7d'
fail = []


def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f' - {detail}' if detail else ''))
    if not ok:
        fail.append(name)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def same_json(a, b, ignore=('created', 'generated', 'timestamp', 'out', 'path', 'zip_path', 'inputs')):
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items() if not any(i in k.lower() for i in ignore)}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, float):
            return round(x, 9) if math.isfinite(x) else str(x)
        return x
    return clean(json.loads(Path(a).read_text())) == clean(json.loads(Path(b).read_text()))


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--zip', required=True, type=Path)
ap.add_argument('--plan-dir', type=Path)
a = ap.parse_args()
zipf = a.zip.resolve()

check('frozen archive hash', sha(zipf) == FROZEN, zipf.name)
lines = [l.split(None, 1) for l in (RAW / 'SHA256SUMS.txt').read_text().splitlines() if l.strip()]
bad = [f for h, f in lines if sha(RAW / f.strip()) != h]
check('raw/ files match SHA256SUMS.txt', not bad, f'{len(lines) - len(bad)}/{len(lines)} files')

with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    sel = HERE.parent / 'results' / 'data' / 'STAGE_B_SELECTION.json'

    def run(name, script, *args, ok_codes=(0,)):
        r = subprocess.run([sys.executable, '-I', str(CODE / script), *map(str, args)], capture_output=True, text=True)
        if r.returncode not in ok_codes:
            check(f'{name} runs', False, (r.stderr or r.stdout).strip().splitlines()[-1:])
            return False
        return True

    # G0-D: independent recomputation. Exit code 3 = its built-in comparison flags 165 sign changes vs 164 strict
    # reversals on E+; the paper reports both, so only the recomputed metrics are compared here.
    if run('G0-D', 'G0D/VersionShift_v2_2_G0D_recompute.py', '--zip', zipf, '--out', t / 'G0D', ok_codes=(0, 3)):
        check('G0-D recomputed point estimates', same_json(t / 'G0D/G0D_metrics.json', RAW / 'V2_2_G0D/G0D_metrics.json'))
    if run('G2 A/B/C', 'G2_COMPARE/G2_compare_ABC.py', '--zip', zipf, '--restored', RAW / 'V2_2_G2_PILOT/G2_FOLD_SCORES.csv',
           '--summary', RAW / 'V2_2_G2_PILOT/G2_SUMMARY.json', '--out', t / 'G2_ABC'):
        check('G2 A/B/C pilot comparison', same_json(t / 'G2_ABC/G2_ABC_summary.json', RAW / 'V2_2_G2_ABC/G2_ABC_summary.json')
              and same_json(t / 'G2_ABC/G2_ABC_dataset_details.json', RAW / 'V2_2_G2_ABC/G2_ABC_dataset_details.json'))
    if run('G2-D', 'G2D/g2d_verify_full63.py', '--zip', zipf, '--pilot', RAW / 'V2_2_G2_PILOT/G2_FOLD_SCORES.csv',
           '--holdout', RAW / 'V2_2_G2_HOLDOUT47/G2_FOLD_SCORES.csv', '--out', t / 'G2_FULL63'):
        v = json.loads((t / 'G2_FULL63/G2D_full63_verification.json').read_text())
        check('G2-D restoration, full 63 tasks', same_json(t / 'G2_FULL63/G2D_full63_verification.json', RAW / 'V2_2_G2_FULL63/G2D_full63_verification.json')
              and filecmp.cmp(t / 'G2_FULL63/G2D_dataset_comparisons.csv', RAW / 'V2_2_G2_FULL63/G2D_dataset_comparisons.csv', shallow=False))
    if run('G2-F', 'G2F/g2f_verify_full63.py', '--zip', zipf, '--pilot', RAW / 'V2_2_G2E_PILOT', '--holdout', RAW / 'V2_2_G2F_HOLDOUT47', '--out', t / 'G2F'):
        check('G2-F boosting interventions, full 63 tasks', same_json(t / 'G2F/G2F_full63_verification.json', RAW / 'V2_2_G2F_FULL63/G2F_full63_verification.json')
              and filecmp.cmp(t / 'G2F/G2F_full63_dataset_changes.csv', RAW / 'V2_2_G2F_FULL63/G2F_full63_dataset_changes.csv', shallow=False))
    if run('G2-G', 'G2G/g2g_analyze.py', '--zip', zipf, '--pilot', RAW / 'V2_2_G2E_PILOT', '--holdout', RAW / 'V2_2_G2F_HOLDOUT47', '--out', t / 'G2G'):
        check('G2-G boosting synthesis', same_json(t / 'G2G/G2G_analysis.json', RAW / 'V2_2_G2G_ANALYSIS/G2G_analysis.json')
              and filecmp.cmp(t / 'G2G/G2G_dataset_comparisons.csv', RAW / 'V2_2_G2G_ANALYSIS/G2G_dataset_comparisons.csv', shallow=False))
    if run('G3-A/B', 'G3_AB/g3_robustness.py', '--zip', zipf, '--out', t / 'G3_AB', '--bootstraps', 10000, '--seed', 20261108):
        check('G3-A/B bootstrap and winner margins', same_json(t / 'G3_AB/G3_AB_results.json', RAW / 'V2_2_G3_AB/G3_AB_results.json')
              and all(filecmp.cmp(t / 'G3_AB' / f, RAW / 'V2_2_G3_AB' / f, shallow=False) for f in ('G3_AB_winner_thresholds.csv', 'G3_AB_dataset_metrics.csv')))

    # G3-E: the combined HGB file must be exactly the pilot + hold-out (R2) fold records, then the six-estimator analysis
    key = lambda r: (r['release'], r['dataset'], int(float(r['rep'])), int(float(r['fold'])))
    comb = {key(r): (r['pred_sha'], r['ba']) for r in csv.DictReader(open(RAW / 'V2_2_G3E_HGB_FULL63.csv'))}
    parts = {key(r): (r['pred_sha'], r['ba']) for d in ('V2_2_G3E_PILOT', 'V2_2_G3E_HOLDOUT47_R2')
             for r in csv.DictReader(open(RAW / d / 'G3E_HGB_FOLDS.csv'))}
    check('E2 HGB fold file = pilot + hold-out runs', comb == parts, f'{len(comb)} folds')
    if run('G3-E', 'G3E/g3e_analyze.py', '--zip', zipf, '--hgb', RAW / 'V2_2_G3E_HGB_FULL63.csv', '--selection', sel, '--out', t / 'G3E'):
        r = json.loads((t / 'G3E/G3E_six_estimator_results.json').read_text())
        check('G3-E six-estimator D vs E2', same_json(t / 'G3E/G3E_six_estimator_results.json', RAW / 'V2_2_G3E_ANALYSIS/G3E_six_estimator_results.json'))
    workers = {json.loads((RAW / d / 'RUN_LOCK.json').read_text())['worker_sha256'] for d in ('V2_2_G3E_PILOT', 'V2_2_G3E_HOLDOUT47_R2')}
    check('E2 runs used code/G3E/worker.py', workers == {sha(CODE / 'G3E/worker.py')}, next(iter(workers))[:16] + '...')

    if a.plan_dir:
        if run('G3-G', 'G3G/g3g_verify_local.py', '--plan-dir', a.plan_dir, '--pilot', RAW / 'V2_2_G3E_PILOT',
               '--holdout', RAW / 'V2_2_G3E_HOLDOUT47_R2', '--out', t / 'G3G.json'):
            g = json.loads((t / 'G3G.json').read_text())
            check('G3-G replay of E2 artifacts', g['status'] == 'PASS', f"{g['jobs_verified']} jobs, {g['folds_replayed']} folds")
    else:
        print('[SKIP] G3-G replay (needs --plan-dir with the rebuilt datasets)')

print('\nALL CHECKS PASSED' if not fail else f'\n{len(fail)} CHECK(S) FAILED: {fail}')
sys.exit(1 if fail else 0)
