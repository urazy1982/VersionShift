#!/usr/bin/env python3
"""Check the shipped results without re-running any experiment (a few seconds, read-only).

    python3 paper/verify_results.py

1. File integrity: results/ against BUNDLE_SHA256_MANIFEST.json, project/ against SHA256_MANIFEST_V225.json,
   pipeline/ against SHA256SUMS.txt.
2. A/A controls: in every A/A run the two executions must give bitwise-identical predictions
   (same prediction hash and balanced accuracy on every fold).
3. Independent recomputation: runs pipeline/addons/vs_verify_independent.py on the fold-level files and
   compares every frame (FULL63, HOLDOUT47, PILOT16 x both upgrades) with the recorded VERIFY_SUMMARY.json.
"""
import hashlib, json, math, subprocess, sys, tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES, PROJ, PIPE = ROOT / 'results', ROOT / 'project', ROOT / 'pipeline'
fail = []


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f' - {detail}' if detail else ''))
    if not ok:
        fail.append(name)


# 1. integrity
m = json.loads((RES / 'BUNDLE_SHA256_MANIFEST.json').read_text())['files']
bad = [p for p, h in m.items() if not (RES / p).is_file() or sha(RES / p) != (h if isinstance(h, str) else h['sha256'])]
check('results/ match the Phase 1 bundle manifest', not bad, f'{len(m) - len(bad)}/{len(m)} files')
m = json.loads((PROJ / 'SHA256_MANIFEST_V225.json').read_text())['files']
bad = [p for p, h in m.items() if not (PROJ / p).is_file() or sha(PROJ / p) != h]
check('project/ matches the frozen v2.2.5 manifest', not bad, f'{len(m) - len(bad)}/{len(m)} files')
lines = [l.split() for l in (PIPE / 'SHA256SUMS.txt').read_text().splitlines() if l.strip()]
bad = [f for h, f in lines if not (PIPE / f.lstrip('*')).is_file() or sha(PIPE / f.lstrip('*')) != h]
check('pipeline/ matches the v2.2.6 package checksums', not bad, f'{len(lines) - len(bad)}/{len(lines)} files')

# 2. A/A controls from the fold-level files
for run in sorted((RES / 'STAGE_B_RUNS').glob('AA_*')):
    d = pd.read_csv(run / 'CROSS_RELEASE_FOLDS.csv.gz', dtype={'release': str})
    d = d[d.status == 'ok']
    k = ['dataset', 'algo', 'policy', 'rep', 'fold']
    j = d[d.condition == 'AA1'].set_index(k)[['pred_sha', 'ba']].join(
        d[d.condition == 'AA2'].set_index(k)[['pred_sha', 'ba']], rsuffix='_2', how='outer')
    jobs = j.reset_index().groupby(['dataset', 'algo']).ngroups
    same = (not j.isna().any().any()) and bool((j.pred_sha == j.pred_sha_2).all()) and bool((j.ba == j.ba_2).all())
    check(f'A/A {run.name}: two executions bitwise identical', same, f'{len(j)} folds, {jobs} jobs')

# 3. independent recomputation, in the directory layout the tool expects
with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    for name in ('STAGE_B_RUNS', 'STAGE_C_CPU_COMPLETION'):
        (t / name).symlink_to(RES / name)
    for f in (RES / 'data').glob('*.json'):
        (t / f.name).symlink_to(f)
    out = t / 'out'
    subprocess.run([sys.executable, '-I', str(PIPE / 'addons' / 'vs_verify_independent.py'), '--data', str(t), '--out', str(out)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    new = json.loads((out / 'VERIFY_SUMMARY.json').read_text())
ref = json.loads((RES / 'INDEPENDENT_VERIFICATION' / 'VERIFY_SUMMARY.json').read_text())


def close(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


for key, r in ref['frames'].items():
    check(f'recomputed {key}', close(new['frames'].get(key), r),
          f"M = {r['M_pp']:.4f} pp, best classifier changes {r['O1_best_algorithm_changes']}/{r['K_complete']}")
for key, r in ref['runs'].items():
    if r.get('kind') != 'AA':  # A/A raw per-job files are not shipped; checked in step 2 instead
        check(f'recomputed run {key}', close(new['runs'].get(key), r))

print('\nALL CHECKS PASSED' if not fail else f'\n{len(fail)} CHECK(S) FAILED: {fail}')
sys.exit(1 if fail else 0)
