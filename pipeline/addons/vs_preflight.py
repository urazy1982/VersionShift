#!/usr/bin/env python3
"""VersionShift WSL package - read-only pre-flight check (changes nothing).

Checks everything the CPU completion kit and the add-ons depend on before any job starts:
frozen source hashes, plan cache, Stage B evidence, historical interpreters, controller
venv, disk, CPU, memory. Writes PREFLIGHT_REPORT.json into the data directory.
Exit code 0 = all required checks pass; 1 = at least one FAIL.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

EXPECTED_PLAN = '523c42a3dabee44e66d43d538b228939b01fd7fe1b68285f542f0980906492f4'
EXPECTED_B = '55554df012a30d7000d3b5840babd23f8e10b9daca57b10e8cf6099fce92943c'
EXPECTED_ORACLE = '33f04fb6c6da89654f3524cfaef290644b80e37afcd5148dcffa740e9bcfe2f6'
EXPECTED_WORKER = '568dae949cf3ecdef26315e41ef41f36703260ef2484b9bc4e6cc0aa70345dbf'
EXPECTED_CONTROLLER = 'ae4124d63141a5fe4c62ae84a13bf971f6842312986248c9ff79fbd1c313ba23'
STAGE_B_RUNS = {  # name: (policies, pin_from, planned jobs)
    'EDGE_0213_0222': ({'D', 'E', 'P', 'S'}, '0.21.3', 768),
    'EDGE_0213_0222_REVERSE': ({'P'}, '0.22.2.post1', 192),
    'EDGE_180_190': ({'D', 'E', 'P', 'S'}, '1.8.0', 768),
    'EDGE_180_190_REVERSE': ({'P'}, '1.9.1', 192),
}
PHASE_RELEASES = {
    'phase1': ['0.21.3', '0.22.2.post1', '1.8.0', '1.9.1'],
    'phase2': ['0.21.3', '0.22.2.post1', '0.23.2', '0.24.2', '1.0.2', '1.1.3', '1.2.2',
               '1.3.2', '1.4.2', '1.5.2', '1.6.1', '1.7.2', '1.8.0', '1.9.1'],
    'phase3': ['1.4.2-fixedstack', '1.5.2-fixedstack'],
}
results = []


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def check(name, ok, detail='', required=True):
    level = 'PASS' if ok else ('FAIL' if required else 'WARN')
    results.append(dict(check=name, result=level, detail=detail))
    print(f'[{level}] {name}' + (f' - {detail}' if detail else ''), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='/mnt/c/VersionShift_v2_2_5')
    ap.add_argument('--data', default=str(Path.home() / 'versionshift_data'))
    ap.add_argument('--envs', default=os.environ.get('VS_ENV_ROOT', str(Path.home() / 'versionshift_envs')))
    ap.add_argument('--phase', choices=sorted(PHASE_RELEASES), default='phase1')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    pr, dat, envs = Path(a.project), Path(a.data), Path(a.envs)

    # 1. Project source integrity
    check('project directory exists', pr.is_dir(), str(pr))
    man = pr / 'SHA256_MANIFEST_V225.json'
    if check('v2.2.5 SHA256 manifest present', man.is_file(), str(man)):
        files = json.loads(man.read_text())['files']
        bad = [p for p, h in files.items() if not (pr / p).is_file() or sha(pr / p) != h]
        check('all v2.2.5 manifest files unchanged', not bad,
              f'{len(files) - len(bad)}/{len(files)} match' + (f'; differ/missing: {bad[:8]}' if bad else ''))
    for rel, h in (('crossversion/worker.py', EXPECTED_WORKER), ('crossversion/controller.py', EXPECTED_CONTROLLER)):
        p = pr / rel
        check(f'{rel} matches Stage B run locks', p.is_file() and sha(p) == h)
    for rel in ('code/analysis_v221.py', 'code/lenses_crossrelease_v225.py', 'code/emulate_plan.py',
                'crossversion/build_envs.sh'):
        check(f'{rel} present', (pr / rel).is_file())
    oc = pr / 'ORACLE_FRAME_REVIEWED.csv'
    check('oracle frame hash', oc.is_file() and sha(oc) == EXPECTED_ORACLE)
    of = pr / 'ORACLE_FROZEN.json'
    if check('ORACLE_FROZEN.json present', of.is_file()):
        fz = json.loads(of.read_text())
        fp = Path(fz.get('frame_path', ''))
        if not fp.is_absolute():
            fp = pr / fp
        check('ORACLE_FROZEN status and frame path', fz.get('status') == 'FROZEN_COMPLETE' and fp.is_file()
              and sha(fp) == EXPECTED_ORACLE, f'frame_path={fp}')

    # 2. Plan cache: PLAN.json hash and every dataset/fold artifact hash
    plan = dat / 'STAGE_B_PLAN_CACHED'
    pj = plan / 'PLAN.json'
    if check('PLAN.json hash', pj.is_file() and sha(pj) == EXPECTED_PLAN, str(pj)):
        doc = json.loads(pj.read_text())
        bad = []
        for d in doc['datasets']:
            for f, k in (('dataset_npz', 'dataset_sha256'), ('folds_npz', 'folds_sha256')):
                p = plan / d[f]
                if not p.is_file() or sha(p) != d[k]:
                    bad.append(d[f])
        check('all 63 plan datasets and folds present with matching hashes',
              len(doc['datasets']) == 63 and not bad,
              f"{len(doc['datasets'])} datasets; bad/missing: {bad[:6]}" if bad else f"{len(doc['datasets'])} datasets")
    sel = dat / 'STAGE_B_SELECTION.json'
    check('STAGE_B_SELECTION.json hash', sel.is_file() and sha(sel) == EXPECTED_B, str(sel))

    # 3. Stage B evidence that the kit merges with Stage C
    runs = dat / 'STAGE_B_RUNS'
    for name, (pols, pin, n) in STAGE_B_RUNS.items():
        root = runs / name
        try:
            lock = json.loads((root / 'RUN_LOCK.json').read_text())
            summ = json.loads((root / 'COLLECT_SUMMARY.json').read_text())
            ok = (lock['plan_sha256'] == EXPECTED_PLAN and lock.get('selection_sha256') == EXPECTED_B
                  and lock.get('worker_sha256') == EXPECTED_WORKER and set(lock['policies']) == pols
                  and lock['pin_from'] == pin and summ.get('planned_jobs') == n and not summ.get('missing_jobs')
                  and sum(summ['status_counts'].values()) == n and (root / 'CROSS_RELEASE_FOLDS.csv.gz').is_file())
            check(f'Stage B run {name} complete and bound', ok, f'{n} jobs')
        except Exception as e:  # noqa
            check(f'Stage B run {name} complete and bound', False, repr(e))
    check('existing A/A run AA_0213 present (kept; kit re-runs 0.21.3 A/A as AA_0_21_3)',
          (runs / 'AA_0213' / 'RUN_LOCK.json').is_file(), required=False)

    # 4. Kit outputs that must not pre-exist without a lock
    for name in ('STAGE_C47_POS', 'STAGE_C47_POS_REVERSE', 'STAGE_C47_NEG', 'STAGE_C47_NEG_REVERSE'):
        root = runs / name
        check(f'{name}: absent or resumable', (not root.exists()) or (root / 'RUN_LOCK.json').is_file(),
              'exists with lock: will RESUME' if root.exists() else 'will be created')

    # 5. Historical interpreters needed by this phase
    for rel in PHASE_RELEASES[a.phase]:
        py = envs / f'sk-{rel}' / 'bin' / 'python'
        want = rel.replace('-fixedstack', '')
        if not py.is_file():
            check(f'interpreter sk-{rel}', False, f'missing {py}' + (' (built in this phase)' if a.phase != 'phase1' else ''),
                  required=(a.phase == 'phase1'))
            continue
        try:
            out = subprocess.run([str(py), '-c', 'import sklearn,numpy,scipy,sys;print(sklearn.__version__,numpy.__version__,scipy.__version__,sys.version.split()[0])'],
                                 capture_output=True, text=True, timeout=120).stdout.split()
            check(f'interpreter sk-{rel}', out and out[0] == want,
                  f'sklearn {out[0]} numpy {out[1]} scipy {out[2]} python {out[3]}' if len(out) == 4 else 'import failed')
        except Exception as e:  # noqa
            check(f'interpreter sk-{rel}', False, repr(e))
        check(f'sk-{rel} FREEZE.txt and PROVENANCE.json', (py.parent.parent / 'FREEZE.txt').is_file()
              and (py.parent.parent / 'PROVENANCE.json').is_file(), required=False)
    if a.phase == 'phase2':
        check('uv available for build_envs.sh', shutil.which('uv') is not None)
        check('VS_PY37 set (needed only if 0.21.3 env must be rebuilt)', bool(os.environ.get('VS_PY37')), required=False)

    # 6. Controller interpreter (the one running this script)
    try:
        import numpy, pandas, scipy  # noqa
        check('controller python has numpy/pandas/scipy', True,
              f'{sys.executable}: numpy {numpy.__version__} pandas {pandas.__version__} scipy {scipy.__version__}')
    except ImportError as e:
        check('controller python has numpy/pandas/scipy', False, repr(e))
    check('running inside ~/versionshift_controller venv', 'versionshift_controller' in sys.prefix, sys.prefix, required=False)

    # 7. Machine resources
    free = shutil.disk_usage(dat if dat.exists() else Path.home()).free / 1e9
    check('free disk in data directory >= 5 GB', free >= 5, f'{free:.1f} GB')
    cpus = os.cpu_count() or 1
    check(f'CPUs >= workers ({a.workers})', cpus >= a.workers, f'{cpus} logical CPUs')
    try:
        mem = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
        gb = mem['MemAvailable'] / 1e6
        check('available memory >= 16 GB', gb >= 16, f'{gb:.1f} GB available', required=False)
    except Exception:  # noqa
        pass
    thr = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    check('BLAS/OpenMP thread variables set to 1', all(v == '1' for v in thr.values()), str(thr))
    check('project on /mnt/c (as ORACLE_FROZEN expects)', str(pr).startswith('/mnt/c/'), required=False)

    fails = [r for r in results if r['result'] == 'FAIL']
    report = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), phase=a.phase, project=str(pr), data=str(dat),
                  envs=str(envs), n_fail=len(fails), n_warn=sum(r['result'] == 'WARN' for r in results),
                  checks=results)
    if dat.is_dir():
        (dat / f'PREFLIGHT_REPORT_{a.phase}.json').write_text(json.dumps(report, indent=2))
    print(f"\nPREFLIGHT {a.phase}: {'OK' if not fails else 'FAILED'} - {len(fails)} fail, {report['n_warn']} warn")
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
