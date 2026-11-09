#!/usr/bin/env python3
"""Freeze AMENDMENT_V226_PRE_RUN.md into the data directory (idempotent, write-once).

Records the amendment SHA-256, the freeze time, and which kit run directories already
existed (and how many results they held) at freeze time, so a reader can see whether the
amendment preceded the Stage C / longitudinal / fixed-stack outcomes.
"""
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path

WATCH = ('STAGE_C47_POS', 'STAGE_C47_POS_REVERSE', 'STAGE_C47_NEG', 'STAGE_C47_NEG_REVERSE',
         'AA_0_21_3', 'AA_0_22_2_post1', 'AA_1_8_0', 'AA_1_9_1',
         'LONGITUDINAL14_D_E_HOLDOUT47', 'FIXEDSTACK_142_152_D_E_HOLDOUT47')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amendment', required=True)
    ap.add_argument('--data', default=str(Path.home() / 'versionshift_data'))
    a = ap.parse_args()
    src, dat = Path(a.amendment), Path(a.data)
    dest = dat / 'AMENDMENT_V226_PRE_RUN.md'
    rec = dat / 'AMENDMENT_V226_FREEZE.json'
    h = sha(src)
    if rec.exists():
        old = json.loads(rec.read_text())
        if old['amendment_sha256'] != h or sha(dest) != h:
            sys.exit('STOP: a different amendment is already frozen in ' + str(rec) + '; do not overwrite')
        print('AMENDMENT ALREADY FROZEN', old['frozen_at'], 'SHA-256', h)
        return
    runs = dat / 'STAGE_B_RUNS'
    state = {}
    for name in WATCH:
        r = runs / name / 'results'
        state[name] = (len(list(r.glob('*.json'))) if r.is_dir() else None)
    started = {k: v for k, v in state.items() if v}
    shutil.copyfile(src, dest)
    doc = dict(amendment='AMENDMENT_V226_PRE_RUN.md', amendment_sha256=h,
               frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               result_files_at_freeze=state,
               preceded_all_new_outcomes=not started,
               note='External timestamp still required: send this SHA-256 off-machine (e-mail/GitHub/OSF) before Phase 1.')
    rec.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    print('AMENDMENT FROZEN', doc['frozen_at'])
    print('SHA-256', h)
    if started:
        print('WARNING: some runs already had results at freeze time:', started)
        print('         the amendment does NOT precede those outcomes; this is recorded in', rec)
    print('NEXT: record this SHA-256 outside this machine (e-mail / git push / OSF) before starting Phase 1.')


if __name__ == '__main__':
    main()
