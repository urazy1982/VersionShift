"""VersionShift v2.1 cross-release controller (runs in the modern controller environment).

  freeze     freeze datasets + fold indices + seeds into an immutable plan, with an
             EXCLUSION LEDGER covering every task of the sampling frame
  introspect dump each release's resolved constructor defaults (objective default-change detector)
  run        execute jobs in each frozen env; idempotent and resumable (--resume)
  collect    completeness check of every planned job/fold + long table for analysis_v21
  replay     independently recompute every score from stored prediction bytes

Releases are name=path/to/python pairs. --aa allows two names to share one interpreter
(A/A self-test: predictions must be bitwise identical).
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

ALGOS = ('LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB')
HERE = Path(__file__).resolve().parent


def h(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4], 'big') % (2**31 - 1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'))


# ------------------------------------------------------------------ freeze
def encode_frame(X, categorical_mask):
    """Central, label-free encoding: one-hot for categoricals (missing -> own column), numeric kept;
    NaN left in place for the worker's train-fold median imputation."""
    import pandas as pd
    X = pd.DataFrame(X).copy()
    cols = []
    for j, c in enumerate(X.columns):
        if categorical_mask[j]:
            d = pd.get_dummies(X[c].astype('object').where(X[c].notna(), '__missing__'), prefix=str(c), dtype=float)
            cols.append(d)
        else:
            cols.append(pd.to_numeric(X[c], errors='coerce').astype(float).to_frame())
    return pd.concat(cols, axis=1).to_numpy(dtype=np.float64)


def frame_bundled():
    from sklearn import datasets as skd
    for name, loader in [('iris', skd.load_iris), ('wine', skd.load_wine), ('breast_cancer', skd.load_breast_cancer)]:
        X, y = loader(return_X_y=True)
        yield dict(name=name, origin='sklearn-bundled'), X, y, [False] * X.shape[1]


def frame_cc18():
    import openml   # controller env only
    suite = openml.study.get_suite(99)
    for tid in suite.tasks:
        try:
            task = openml.tasks.get_task(tid, download_splits=False); ds = task.get_dataset()
            X, y, cat, _ = ds.get_data(target=ds.default_target_attribute)
            yield dict(name='openml_%d' % tid, origin='OpenML-CC18', task=int(tid), dataset_id=int(ds.dataset_id),
                       dataset_name=ds.name, version=int(ds.version)), X, y, list(cat)
        except Exception as e:
            yield dict(name='openml_%d' % tid, origin='OpenML-CC18', task=int(tid), load_error=repr(e)), None, None, None


def freeze(args):
    out = Path(args.plan_dir)
    if out.exists():
        raise SystemExit('plan directory exists; plans are immutable')
    out.mkdir(parents=True)
    from sklearn.model_selection import StratifiedKFold
    plan = dict(schema='versionshift-v2.1-plan', created=time.strftime('%Y-%m-%dT%H:%M:%S'), reps=args.reps, folds=5,
                rules=dict(max_n=args.max_n, min_class_count=10, categorical='central one-hot', missing='train-fold median in worker',
                           large_n='stratified subsample to max_n (seeded)'), datasets=[])
    ledger = []
    frame = frame_bundled() if args.source == 'bundled' else frame_cc18()
    for info, X, y, cat in frame:
        entry = dict(info, status='included', reasons=[])
        if X is None:
            entry.update(status='excluded', reasons=['load_error']); ledger.append(entry); continue
        Xn = encode_frame(X, cat) if any(cat) else np.asarray(X, dtype=np.float64)
        _, yy = np.unique(np.asarray(y).astype(str), return_inverse=True)
        entry.update(n_raw=int(len(yy)), p_raw=int(np.shape(X)[1]), n_categorical=int(sum(cat)),
                     missing_cells=int(np.isnan(Xn).sum()), p_encoded=int(Xn.shape[1]))
        counts = np.bincount(yy)
        if counts.min() < 10:
            entry['status'] = 'excluded'; entry['reasons'].append('class_with_fewer_than_10_rows')
        if Xn.shape[1] > args.max_p:
            entry['status'] = 'excluded'; entry['reasons'].append('more_than_%d_encoded_features' % args.max_p)
        if entry['status'] == 'included' and len(yy) > args.max_n:
            from sklearn.model_selection import train_test_split
            keep, _ = train_test_split(np.arange(len(yy)), train_size=args.max_n, stratify=yy, random_state=h('subsample', info['name']))
            Xn, yy = Xn[np.sort(keep)], yy[np.sort(keep)]; entry['reasons'].append('subsampled_to_%d' % args.max_n)
        ledger.append(entry)
        if entry['status'] != 'included':
            continue
        name = info['name']
        dpath = out / (name + '.npz'); np.savez_compressed(dpath, X=Xn, y=yy.astype(np.int64))
        fold_arrays, seeds, keys = {}, {}, []
        for r in range(args.reps):
            skf = StratifiedKFold(5, shuffle=True, random_state=h('split', name, r))
            for f, (tr, te) in enumerate(skf.split(Xn, yy)):
                key = 'r%d_f%d' % (r, f); keys.append(key)
                fold_arrays[key + '_train'] = tr.astype(np.int64); fold_arrays[key + '_test'] = te.astype(np.int64)
                seeds[key] = h('model', name, r, f)
        fpath = out / (name + '_folds.npz'); np.savez_compressed(fpath, **fold_arrays)
        plan['datasets'].append(dict(name=name, info=info, n=int(len(yy)), p=int(Xn.shape[1]), k=int(yy.max() + 1),
                                     dataset_npz=dpath.name, dataset_sha256=sha(dpath), folds_npz=fpath.name,
                                     folds_sha256=sha(fpath), fold_keys=keys, seeds=seeds,
                                     test_sizes={k_: int(len(fold_arrays[k_ + '_test'])) for k_ in keys}))
    (out / 'EXCLUSION_LEDGER.json').write_text(json.dumps(ledger, indent=1, sort_keys=True, default=str))
    plan['ledger_sha256'] = sha(out / 'EXCLUSION_LEDGER.json')
    plan['frame_size'] = len(ledger); plan['eligible'] = len(plan['datasets'])
    (out / 'PLAN.json').write_text(json.dumps(plan, indent=1, sort_keys=True, default=str))
    print('frame %d, eligible %d; PLAN sha256 %s' % (len(ledger), len(plan['datasets']), sha(out / 'PLAN.json')))


# ------------------------------------------------------------------ introspect
def introspect(args):
    releases = dict(x.split('=', 1) for x in args.release)
    code = ('import json,sys;sys.path.insert(0,%r);import worker as w;c=w.classes();'
            'print(json.dumps({"provenance":w.provenance(),"defaults":{a:w.defaults_of(c[a]) for a in c}}))' % str(HERE))
    out = {}
    for rel, py in releases.items():
        out[rel] = json.loads(subprocess.run([py, '-I', '-c', code], capture_output=True, text=True, check=True).stdout)
    names = list(releases)
    diffs = []
    for a, b in zip(names, names[1:]):
        for algo in ALGOS:
            da, db = out[a]['defaults'][algo], out[b]['defaults'][algo]
            for k in sorted(set(da) | set(db)):
                if da.get(k, '<absent>') != db.get(k, '<absent>'):
                    diffs.append(dict(edge='%s->%s' % (a, b), estimator=algo, param=k,
                                      old=da.get(k, '<absent>'), new=db.get(k, '<absent>')))
    Path(args.out).write_text(json.dumps(dict(releases=out, adjacent_default_diffs=diffs), indent=1, sort_keys=True))
    print('default diffs on adjacent edges:', len(diffs))


# ------------------------------------------------------------------ run
def version_of(py):
    return subprocess.run([py, '-I', '-c', 'import sklearn;print(sklearn.__version__)'], capture_output=True,
                          text=True, check=True).stdout.strip()


def record_valid(rec_path):
    try:
        rec = json.loads(Path(rec_path).read_text())
    except Exception:
        return False
    if rec.get('status') == 'ok':
        npz = Path(str(rec_path)[:-5] + '.npz')
        return npz.is_file() and sha(npz) == rec.get('npz_sha256')
    return rec.get('status') in ('not_portable', 'fit_failed')


def run(args):
    plan_dir = Path(args.plan_dir); plan = json.loads((plan_dir / 'PLAN.json').read_text())
    out = Path(args.out); releases = dict(x.split('=', 1) for x in args.release)
    worker = str(HERE / 'worker.py')
    if out.exists() and not args.resume:
        raise SystemExit('output exists; use --resume to continue an interrupted run (never overwrites)')
    lock_path = out / 'RUN_LOCK.json'
    lock = dict(plan_sha256=sha(plan_dir / 'PLAN.json'), worker_sha256=sha(worker), controller_sha256=sha(__file__),
                releases={k: dict(python=v, sklearn=version_of(v)) for k, v in releases.items()},
                policies=args.policies.split(','), algorithms=args.algorithms.split(','),
                translations_file=args.translations, pin_from=args.pin_from)
    if args.translations:
        lock['translations'] = json.loads(Path(args.translations).read_text())
    if len({v['sklearn'] for v in lock['releases'].values()}) < len(releases) and not args.aa:
        raise SystemExit('two releases resolve to the same scikit-learn version (use --aa only for the A/A self-test)')
    if set(lock['policies']) & {'P', 'PT', 'S'}:
        if args.pin_from not in releases:
            raise SystemExit('policies P/PT/S need --pin-from <release name>')
        code = ('import json,sys;sys.path.insert(0,%r);import worker as w;c=w.classes();'
                'print(json.dumps({a:w.defaults_of(c[a]) for a in c}))' % str(HERE))
        lock['pinned'] = json.loads(subprocess.run([releases[args.pin_from], '-I', '-c', code], capture_output=True,
                                                   text=True, check=True).stdout)
    selected = None
    if args.selection:
        selection = json.loads(Path(args.selection).read_text())
        if selection.get('plan_sha256') != sha(plan_dir / 'PLAN.json'):
            raise SystemExit('selection was made for a different frozen plan')
        selected = set(selection['stage_b_datasets'])
        available = {d['name'] for d in plan['datasets']}
        if not selected or not selected <= available or len(selected) != len(selection['stage_b_datasets']):
            raise SystemExit('selection contains missing or duplicated dataset IDs')
        lock['selection_sha256'] = sha(args.selection)
        lock['selected_datasets'] = sorted(selected)
    if lock_path.exists():
        old = json.loads(lock_path.read_text())
        if canonical(old) != canonical(lock):
            raise SystemExit('resume refused: plan, sources, releases or policies differ from the original RUN_LOCK')
    else:
        (out / 'jobs').mkdir(parents=True, exist_ok=True); (out / 'results').mkdir(exist_ok=True)
        lock_path.write_text(json.dumps(lock, indent=1, sort_keys=True))
    status_log = open(out / 'JOB_STATUS.jsonl', 'a')
    done = skipped = 0
    for d in plan['datasets']:
        if selected is not None and d['name'] not in selected:
            continue
        for f, key in (('dataset_npz', 'dataset_sha256'), ('folds_npz', 'folds_sha256')):
            if sha(plan_dir / d[f]) != d[key]:
                raise SystemExit('plan artifact changed: ' + d[f])
        for rel, py in releases.items():
            for a in lock['algorithms']:
                for pol in lock['policies']:
                    jid = '%s__%s__%s__%s' % (rel, d['name'], a, pol)
                    rec_path = out / 'results' / (jid + '.json')
                    if rec_path.exists():
                        if record_valid(rec_path):
                            skipped += 1; continue
                        q = out / '_quarantine' / time.strftime('%Y%m%dT%H%M%S'); q.mkdir(parents=True, exist_ok=True)
                        for p in (rec_path, rec_path.with_suffix('.npz')):
                            if p.exists(): shutil.move(str(p), str(q / p.name))
                    for stray in (rec_path.with_suffix('.npz'), Path(str(rec_path.with_suffix('')) + '.npz.tmp.npz')):
                        if stray.exists():
                            q = out / '_quarantine' / 'partial'; q.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(stray), str(q / (stray.name + '.' + str(int(time.time())))))
                    job = dict(id=jid, release=rel, dataset=d['name'], algorithm=a, policy=pol,
                               expected_sklearn=lock['releases'][rel]['sklearn'],
                               dataset_npz=str(plan_dir / d['dataset_npz']), folds_npz=str(plan_dir / d['folds_npz']),
                               fold_keys=d['fold_keys'], seeds=d['seeds'], out_prefix=str(out / 'results' / jid),
                               pinned=lock.get('pinned', {}).get(a) if pol in ('P', 'PT', 'S') else None,
                               translations=lock.get('translations', {}).get(a, {}) if pol == 'PT' else None,
                               pin_source_sklearn=lock['releases'][args.pin_from]['sklearn'] if pol in ('P', 'PT', 'S') else None)
                    job['job_sha256'] = hashlib.sha256(canonical(job).encode()).hexdigest()
                    jp = out / 'jobs' / (jid + '.json'); jp.write_text(json.dumps(job, sort_keys=True))
                    t0 = time.time()
                    r = subprocess.run([py, '-I', '-B', worker, str(jp)], capture_output=True, text=True, timeout=args.timeout)
                    if r.returncode != 0:
                        status_log.write(json.dumps(dict(id=jid, status='worker_crashed', stderr=r.stderr[-400:])) + '\n'); status_log.flush()
                        raise SystemExit('worker crashed (outputs preserved; rerun with --resume): ' + jid)
                    rec = json.loads(rec_path.read_text())
                    status_log.write(json.dumps(dict(id=jid, status=rec['status'], portability=rec.get('portability'),
                                                     seconds=round(time.time() - t0, 2))) + '\n'); status_log.flush()
                    done += 1
    print('jobs executed: %d, already complete: %d' % (done, skipped))


# ------------------------------------------------------------------ collect / replay
def _load(args):
    out = Path(args.out); plan_dir = Path(args.plan_dir)
    lock = json.loads((out / 'RUN_LOCK.json').read_text()); plan = json.loads((plan_dir / 'PLAN.json').read_text())
    if sha(plan_dir / 'PLAN.json') != lock['plan_sha256']:
        raise SystemExit('plan changed since run')
    return out, plan_dir, lock, plan


def collect(args):
    import pandas as pd
    out, plan_dir, lock, plan = _load(args)
    rows, missing = [], []
    for d in plan['datasets']:
        if lock.get('selected_datasets') and d['name'] not in lock['selected_datasets']:
            continue
        for rel in lock['releases']:
            for a in lock['algorithms']:
                for pol in lock['policies']:
                    jid = '%s__%s__%s__%s' % (rel, d['name'], a, pol); p = out / 'results' / (jid + '.json')
                    if not p.exists():
                        missing.append(dict(id=jid, reason='no_record')); continue
                    rec = json.loads(p.read_text())
                    base = dict(condition=rel, release=rel, dataset=d['name'], algo=a, policy=pol, k=d['k'], n=d['n'],
                                status=rec['status'], portability=rec.get('portability'))
                    if rec['status'] != 'ok':
                        rows.append(dict(base, detail=rec.get('detail') or rec.get('error'))); continue
                    npz = p.with_suffix('.npz')
                    if sha(npz) != rec['npz_sha256']:
                        raise SystemExit('prediction artifact tampered: ' + npz.name)
                    if rec['fold_keys'] != d['fold_keys']:
                        raise SystemExit('fold keys differ from plan: ' + jid)
                    z = np.load(npz, allow_pickle=False)
                    for key, ba in zip(rec['fold_keys'], rec['ba']):
                        pred = z[key + '_pred']
                        if len(pred) != d['test_sizes'][key]:
                            raise SystemExit('incomplete prediction array: %s %s' % (jid, key))
                        r, f = key[1:].split('_f')
                        rows.append(dict(base, sklearn=rec['provenance']['sklearn'], numpy=rec['provenance']['numpy'],
                                         rep=int(r), fold=int(f), ba=ba,
                                         pred_sha=hashlib.sha256(pred.tobytes()).hexdigest()[:16]))
    df = pd.DataFrame(rows)
    df.to_csv(out / 'CROSS_RELEASE_FOLDS.csv.gz', index=False)
    summary = dict(planned_jobs=len(lock.get('selected_datasets') or plan['datasets']) * len(lock['releases']) * len(lock['algorithms']) * len(lock['policies']),
                   missing_jobs=missing, status_counts=df.drop_duplicates(['release', 'dataset', 'algo', 'policy']).status.value_counts().to_dict() if len(df) else {},
                   portability_counts=df.drop_duplicates(['release', 'dataset', 'algo', 'policy']).portability.value_counts(dropna=False).to_dict() if len(df) else {})
    (out / 'COLLECT_SUMMARY.json').write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != 'missing_jobs'}, default=str), 'missing:', len(missing))
    return df


def replay(args):
    out, plan_dir, lock, plan = _load(args)
    checked = 0
    for d in plan['datasets']:
        if lock.get('selected_datasets') and d['name'] not in lock['selected_datasets']:
            continue
        y = np.load(plan_dir / d['dataset_npz'], allow_pickle=False)['y']
        folds = np.load(plan_dir / d['folds_npz'], allow_pickle=False)
        k = int(y.max()) + 1
        for p in sorted((out / 'results').glob('*__%s__*.json' % d['name'])):
            rec = json.loads(p.read_text())
            if rec['status'] != 'ok':
                continue
            z = np.load(p.with_suffix('.npz'), allow_pickle=False)
            for key, ba in zip(rec['fold_keys'], rec['ba']):
                te = folds[key + '_test']; pred = z[key + '_pred']
                again = float(np.mean([np.mean(pred[y[te] == c] == c) for c in range(k) if np.any(y[te] == c)]))
                if abs(again - ba) > 1e-12:
                    raise SystemExit('REPLAY MISMATCH %s %s %r != %r' % (rec['id'], key, again, ba))
                checked += 1
    print('REPLAY PASS: %d fold scores recomputed from prediction bytes' % checked)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('freeze'); f.add_argument('--plan-dir', required=True); f.add_argument('--reps', type=int, default=6)
    f.add_argument('--source', default='openml-cc18', choices=['openml-cc18', 'bundled'])
    f.add_argument('--max-n', type=int, default=5000); f.add_argument('--max-p', type=int, default=500)
    i = sub.add_parser('introspect'); i.add_argument('--release', action='append', required=True); i.add_argument('--out', required=True)
    r = sub.add_parser('run'); r.add_argument('--plan-dir', required=True); r.add_argument('--out', required=True)
    r.add_argument('--release', action='append', required=True, help='name=/path/to/python (in release order)')
    r.add_argument('--policies', default='D,E,P'); r.add_argument('--algorithms', default=','.join(ALGOS))
    r.add_argument('--timeout', type=int, default=7200); r.add_argument('--aa', action='store_true')
    r.add_argument('--pin-from', default=None); r.add_argument('--translations', default=None, help='JSON of documented translations (PT arm only)')
    r.add_argument('--resume', action='store_true')
    r.add_argument('--selection', help='frozen STAGE_B_SELECTION.json; execute selected datasets only')
    for name in ('collect', 'replay'):
        c = sub.add_parser(name); c.add_argument('--plan-dir', required=True); c.add_argument('--out', required=True)
    a = ap.parse_args()
    dict(freeze=freeze, introspect=introspect, run=run, collect=collect, replay=replay)[a.cmd](a)


if __name__ == '__main__':
    main()
