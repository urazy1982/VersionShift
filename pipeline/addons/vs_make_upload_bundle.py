#!/usr/bin/env python3
"""Build the results ZIP to upload for independent audit / manuscript drafting.

Default: reports, merged fold CSVs, run locks/summaries/fold CSVs/status logs of every run,
A/A identity files, amendment + freeze record, preflight + verification reports, kit logs,
and environment provenance. Raw per-job prediction artifacts (results/*.npz) are excluded
unless --with-raw is given. A SHA-256 manifest of every included file is added.
"""
import argparse, hashlib, json, time, zipfile
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(Path.home() / 'versionshift_data'))
    ap.add_argument('--envs', default=str(Path.home() / 'versionshift_envs'))
    ap.add_argument('--out-dir', default='/mnt/c/VersionShift_v2_2_5')
    ap.add_argument('--label', default='phase1')
    ap.add_argument('--with-raw', action='store_true')
    a = ap.parse_args()
    dat, envs = Path(a.data), Path(a.envs)
    files = {}

    def add(p, arc):
        p = Path(p)
        if p.is_file():
            files[arc] = p

    for name in ('STAGE_C_HOLDOUT47_SELECTION.json', 'STAGE_C_COMPLETION_PROTOCOL_RECORD.json',
                 'AMENDMENT_V226_PRE_RUN.md', 'AMENDMENT_V226_FREEZE.json', 'STAGE_C_ALL14_INTROSPECTION.json',
                 'STAGE_B_SELECTION.json'):
        add(dat / name, 'data/' + name)
    for p in dat.glob('PREFLIGHT_REPORT_*.json'):
        add(p, 'data/' + p.name)
    for p in dat.glob('*.log'):
        add(p, 'logs/' + p.name)
    add(dat / 'STAGE_B_PLAN_CACHED/PLAN.json', 'data/PLAN.json')
    comp = dat / 'STAGE_C_CPU_COMPLETION'
    for p in comp.rglob('*') if comp.is_dir() else []:
        if p.is_file() and not p.name.endswith('.tmp'):
            add(p, 'STAGE_C_CPU_COMPLETION/' + str(p.relative_to(comp)))
    for p in (dat / 'INDEPENDENT_VERIFICATION').glob('*'):
        add(p, 'INDEPENDENT_VERIFICATION/' + p.name)
    for root in sorted((dat / 'STAGE_B_RUNS').glob('*')):
        if not root.is_dir():
            continue
        for name in ('RUN_LOCK.json', 'COLLECT_SUMMARY.json', 'CROSS_RELEASE_FOLDS.csv.gz', 'AA_IDENTITY.json',
                     'JOB_STATUS.jsonl', 'PARALLEL_JOB_STATUS.jsonl'):
            add(root / name, f'STAGE_B_RUNS/{root.name}/{name}')
        if a.with_raw:
            for p in (root / 'results').glob('*'):
                add(p, f'STAGE_B_RUNS/{root.name}/results/{p.name}')
    for name in ('BUILD_FAILURES.txt', 'UV_VERSION.txt', 'UNAME.txt', 'LSCPU.txt'):
        add(envs / name, 'envs/' + name)
    for d in sorted(envs.glob('sk-*')):
        for name in ('PROVENANCE.json', 'FREEZE.txt', 'lock.txt'):
            add(d / name, f'envs/{d.name}/{name}')

    stamp = time.strftime('%Y%m%d_%H%M')
    out = Path(a.out_dir) / f'VersionShift_Results_{a.label}_{stamp}.zip'
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {arc: sha(p) for arc, p in sorted(files.items())}
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for arc, p in sorted(files.items()):
            z.write(p, arc)
        z.writestr('BUNDLE_SHA256_MANIFEST.json', json.dumps(dict(created=stamp, label=a.label, with_raw=a.with_raw,
                                                                  n_files=len(manifest), files=manifest), indent=2))
    print(f'BUNDLE: {out}  ({len(manifest)} files, {out.stat().st_size / 1e6:.1f} MB)')
    print('Upload this ZIP in the chat.')


if __name__ == '__main__':
    main()
