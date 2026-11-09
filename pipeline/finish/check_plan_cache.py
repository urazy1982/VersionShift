#!/usr/bin/env python3
"""Check a re-built plan cache against the frozen plan used in the study.

Datasets are not redistributed. Rebuild them from OpenML with `controller.py freeze` (folds and seeds are
derived deterministically from dataset names), then check every file against the frozen PLAN.json:

    python3 pipeline/finish/check_plan_cache.py --plan-dir ~/versionshift_data/STAGE_B_PLAN_CACHED

With --install, the frozen PLAN.json is copied into the plan directory once every hash matches, so that the
pre-flight check (which pins the PLAN.json hash) passes.
"""
import argparse, hashlib, json, shutil, sys
from pathlib import Path

FROZEN = Path(__file__).resolve().parents[2] / 'results' / 'data' / 'PLAN.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--plan-dir', required=True, type=Path)
ap.add_argument('--install', action='store_true')
a = ap.parse_args()
plan = json.loads(FROZEN.read_text())
bad, ok = [], 0
for d in plan['datasets']:
    for name_key, sha_key in (('dataset_npz', 'dataset_sha256'), ('folds_npz', 'folds_sha256')):
        if name_key not in d:
            continue
        p = a.plan_dir / Path(d[name_key]).name
        if not p.is_file():
            bad.append(f'missing {p.name}')
        elif sha(p) != d[sha_key]:
            bad.append(f'hash differs {p.name}')
        else:
            ok += 1
print(f'{ok} plan files match the frozen plan; {len(bad)} problems')
for b in bad[:20]:
    print('  ', b)
if bad:
    sys.exit(1)
if a.install:
    shutil.copy2(FROZEN, a.plan_dir / 'PLAN.json')
    print('installed frozen PLAN.json into', a.plan_dir)
