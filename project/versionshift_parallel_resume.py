#!/usr/bin/env python3
"""VersionShift v2.2.5: conservative parallel resume of an existing locked RUN.

Run ONLY after the original sequential controller has exited. Does not edit the
frozen controller/worker/plan. Uses its existing RUN_LOCK.json and job schema.
"""
import argparse
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'))


def valid(recfile, expected_job_hash, worker_hash, expected_fold_keys):
    try:
        rec = json.loads(recfile.read_text())
        if rec.get('job_sha256') != expected_job_hash or rec.get('worker_sha256') != worker_hash:
            return False
        status = rec.get('status')
        if status == 'ok':
            npz = recfile.with_suffix('.npz')
            return (npz.is_file() and sha(npz) == rec.get('npz_sha256') and
                    rec.get('fold_keys') == expected_fold_keys and
                    len(rec.get('ba', [])) == len(expected_fold_keys))
        return status in ('fit_failed', 'not_portable')
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--timeout', type=int, default=7200)
    args = ap.parse_args()
    if not 1 <= args.workers <= 8:
        ap.error('use 1..8 workers; start with 6')
    plan_dir, out = Path(args.plan_dir).resolve(), Path(args.out).resolve()
    project = Path(__file__).resolve().parent
    controller = project / 'crossversion' / 'controller.py'
    worker = project / 'crossversion' / 'worker.py'
    lock = json.loads((out / 'RUN_LOCK.json').read_text())
    plan = json.loads((plan_dir / 'PLAN.json').read_text())
    if sha(plan_dir / 'PLAN.json') != lock['plan_sha256']:
        raise SystemExit('STOP: plan hash mismatch')
    if sha(controller) != lock['controller_sha256'] or sha(worker) != lock['worker_sha256']:
        raise SystemExit('STOP: source hash mismatch; use original v2.2.5 controller and worker')
    if not lock.get('selected_datasets') or not lock.get('selection_sha256'):
        raise SystemExit('STOP: no Stage B selection recorded in RUN_LOCK')
    selected = set(lock['selected_datasets'])
    if not selected <= {d['name'] for d in plan['datasets']}:
        raise SystemExit('STOP: selection not in plan')
    versions = lock['releases']
    for rel, info in versions.items():
        py = info['python']
        if not Path(py).is_file():
            raise SystemExit('STOP: interpreter missing: ' + py)
        found = subprocess.check_output([py, '-I', '-c', 'import sklearn; print(sklearn.__version__)'], text=True).strip()
        if found != info['sklearn']:
            raise SystemExit('STOP: version mismatch on ' + rel)
    # Never run alongside original controller or a second instance of this scheduler.
    pidfile = out / 'PARALLEL_RESUME.pid'
    if pidfile.exists():
        try:
            oldpid = int(pidfile.read_text().strip())
            os.kill(oldpid, 0)
            raise SystemExit('STOP: another parallel scheduler is running, pid=' + str(oldpid))
        except ProcessLookupError:
            pass
    pidfile.write_text(str(os.getpid()))
    log_lock = threading.Lock()
    try:
        tasks = []
        done = []
        for d in plan['datasets']:
            if d['name'] not in selected:
                continue
            for datafield, hashfield in (('dataset_npz', 'dataset_sha256'), ('folds_npz', 'folds_sha256')):
                if sha(plan_dir / d[datafield]) != d[hashfield]:
                    raise SystemExit('STOP: frozen data checksum mismatch: ' + d[datafield])
            for rel, ri in versions.items():
                for a in lock['algorithms']:
                    for pol in lock['policies']:
                        jid = '%s__%s__%s__%s' % (rel, d['name'], a, pol)
                        result = out / 'results' / (jid + '.json')
                        job = dict(id=jid, release=rel, dataset=d['name'], algorithm=a, policy=pol,
                                   expected_sklearn=ri['sklearn'],
                                   dataset_npz=str(plan_dir / d['dataset_npz']),
                                   folds_npz=str(plan_dir / d['folds_npz']),
                                   fold_keys=d['fold_keys'], seeds=d['seeds'],
                                   out_prefix=str(out / 'results' / jid),
                                   pinned=lock.get('pinned', {}).get(a) if pol in ('P', 'PT', 'S') else None,
                                   translations=lock.get('translations', {}).get(a, {}) if pol == 'PT' else None,
                                   pin_source_sklearn=versions[lock['pin_from']]['sklearn'] if pol in ('P', 'PT', 'S') else None)
                        job['job_sha256'] = hashlib.sha256(canon(job).encode()).hexdigest()
                        jobfile = out / 'jobs' / (jid + '.json')
                        if jobfile.exists():
                            try:
                                if json.loads(jobfile.read_text()) != job:
                                    raise SystemExit('STOP: existing job specification differs: ' + jid)
                            except json.JSONDecodeError:
                                raise SystemExit('STOP: corrupt job specification: ' + jid)
                        else:
                            jobfile.write_text(json.dumps(job, sort_keys=True))
                        if result.exists() and valid(result, job['job_sha256'], lock['worker_sha256'], d['fold_keys']):
                            done.append(jid)
                            continue
                        # Fail closed on unexpected/invalid artifacts instead of overwriting.
                        orphan = [p for p in (result, result.with_suffix('.npz'),
                                    Path(str(result.with_suffix('')) + '.npz.tmp.npz'),
                                    Path(str(result) + '.tmp')) if p.exists()]
                        if orphan:
                            raise SystemExit('STOP: invalid/partial artifacts for ' + jid + '; preserve and inspect: ' + str(orphan))
                        tasks.append((jid, ri['python'], str(jobfile)))
        total = len(done) + len(tasks)
        print('Preflight PASS: total=%d already_valid=%d pending=%d workers=%d' % (total, len(done), len(tasks), args.workers), flush=True)
        if total != 768:
            raise SystemExit('STOP: expected 768 jobs, got ' + str(total))
        if not tasks:
            print('ALL JOBS ALREADY COMPLETE')
            return
        status_log = out / 'PARALLEL_JOB_STATUS.jsonl'
        def execute(item):
            jid, py, jp = item
            started = time.monotonic()
            env = os.environ.copy()
            for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
                env[key] = '1'
            try:
                p = subprocess.run([py, '-I', '-B', str(worker), jp],
                                   capture_output=True, text=True, timeout=args.timeout, env=env)
                if p.returncode:
                    return jid, False, 'exit=%d stderr=%s' % (p.returncode, p.stderr[-300:]), time.monotonic()-started
                recpath = out / 'results' / (jid + '.json')
                itemjob = json.loads(Path(jp).read_text())
                if not valid(recpath, itemjob['job_sha256'], lock['worker_sha256'], itemjob['fold_keys']):
                    return jid, False, 'invalid output record or prediction hash', time.monotonic()-started
                return jid, True, json.loads(recpath.read_text())['status'], time.monotonic()-started
            except subprocess.TimeoutExpired:
                return jid, False, 'worker_timeout', time.monotonic()-started
        errors = []
        with futures.ThreadPoolExecutor(max_workers=args.workers) as pool, status_log.open('a') as logfile:
            active = {pool.submit(execute, item): item for item in tasks[:args.workers]}
            next_idx = args.workers
            completed = 0
            while active:
                for future in futures.as_completed(list(active)):
                    item = active.pop(future)
                    jid, ok, status, seconds = future.result()
                    completed += 1
                    record = dict(id=jid, scheduler='v225-resume', status=status, ok=ok, seconds=round(seconds, 2))
                    with log_lock:
                        logfile.write(json.dumps(record, sort_keys=True) + '\n'); logfile.flush()
                        print('[%d/%d] %s %s (%.1fs)' % (completed, len(tasks), jid, status, seconds), flush=True)
                    if not ok:
                        errors.append(jid)
                    if next_idx < len(tasks):
                        newitem = tasks[next_idx]
                        active[pool.submit(execute, newitem)] = newitem
                        next_idx += 1
                    break
        if errors:
            raise SystemExit('STOP: %d worker errors; inspect PARALLEL_JOB_STATUS.jsonl. First: %s' % (len(errors), errors[:3]))
        print('PARALLEL RESUME COMPLETE: %d newly executed; %d previously valid' % (len(tasks), len(done)))
    finally:
        pidfile.unlink(missing_ok=True)

if __name__ == '__main__':
    main()
