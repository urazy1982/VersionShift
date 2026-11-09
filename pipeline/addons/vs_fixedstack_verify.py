#!/usr/bin/env python3
"""Amendment v2.2.6 A4: verify the fixed numerical-stack run before analyze-fixed.

Checks run lock, job inventory, per-row numpy version, and both environments' FREEZE.txt /
PROVENANCE.json (numpy 1.26.4, scipy 1.12.0). Exit 1 on any failure; nothing is substituted.
"""
import argparse, json, re, sys
from pathlib import Path
import pandas as pd

NUMPY, SCIPY = '1.26.4', '1.12.0'
RELEASES = {'1.4.2': 'sk-1.4.2-fixedstack', '1.5.2': 'sk-1.5.2-fixedstack'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(Path.home() / 'versionshift_data'))
    ap.add_argument('--envs', default=str(Path.home() / 'versionshift_envs'))
    a = ap.parse_args()
    dat, envs = Path(a.data), Path(a.envs)
    root = dat / 'STAGE_B_RUNS/FIXEDSTACK_142_152_D_E_HOLDOUT47'
    fails, info = [], {}
    try:
        lock = json.loads((root / 'RUN_LOCK.json').read_text())
        summ = json.loads((root / 'COLLECT_SUMMARY.json').read_text())
        hold = set(json.loads((dat / 'STAGE_C_HOLDOUT47_SELECTION.json').read_text())['stage_b_datasets'])
    except Exception as e:  # noqa
        sys.exit(f'STOP: fixed-stack run not collected or selection missing: {e!r}')
    if set(lock['releases']) != set(RELEASES):
        fails.append(f"releases {list(lock['releases'])}")
    for rel, env in RELEASES.items():
        py = lock['releases'].get(rel, {}).get('python', '')
        if env not in py:
            fails.append(f'{rel} python not from {env}: {py}')
    if set(lock['policies']) != {'D', 'E'}:
        fails.append(f"policies {lock['policies']}")
    if set(lock['selected_datasets']) != hold or len(hold) != 47:
        fails.append('dataset set is not the 47 holdout datasets')
    want = 47 * 2 * 6 * 2
    if summ.get('planned_jobs') != want or summ.get('missing_jobs') or sum(summ.get('status_counts', {}).values()) != want:
        fails.append(f'job inventory {summ}')
    info['status_counts'] = summ.get('status_counts')
    rows = pd.read_csv(root / 'CROSS_RELEASE_FOLDS.csv.gz', dtype=str)
    for rel in RELEASES:
        r = rows[rows.release == rel]
        nv = sorted(r.numpy.dropna().unique())
        sv = sorted(r.sklearn.dropna().unique())
        info[rel] = dict(rows=len(r), numpy=nv, sklearn=sv)
        if nv != [NUMPY]:
            fails.append(f'{rel} fold rows numpy {nv}')
        if sv != [rel]:
            fails.append(f'{rel} fold rows sklearn {sv}')
    for rel, env in RELEASES.items():
        d = envs / env
        try:
            frz = (d / 'FREEZE.txt').read_text().lower()
            if not re.search(rf'^numpy=={re.escape(NUMPY)}\s*$', frz, re.M):
                fails.append(f'{env} FREEZE.txt numpy != {NUMPY}')
            if not re.search(rf'^scipy=={re.escape(SCIPY)}\s*$', frz, re.M):
                fails.append(f'{env} FREEZE.txt scipy != {SCIPY}')
            prov = json.loads((d / 'PROVENANCE.json').read_text())
            info[env] = dict(numpy=prov.get('numpy'), scipy=prov.get('scipy'), sklearn=prov.get('sklearn'))
            if prov.get('numpy') != NUMPY or prov.get('scipy') != SCIPY or prov.get('sklearn') != rel:
                fails.append(f'{env} PROVENANCE {info[env]}')
        except Exception as e:  # noqa
            fails.append(f'{env} provenance unreadable: {e!r}')
    rec = dict(check='Amendment v2.2.6 A4 fixed-stack verification', status='PASS' if not fails else 'FAIL',
               failures=fails, details=info)
    out = dat / 'STAGE_C_CPU_COMPLETION/longitudinal'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'FIXEDSTACK_VERIFICATION_V226.json').write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    sys.exit(0 if not fails else 1)


if __name__ == '__main__':
    main()
