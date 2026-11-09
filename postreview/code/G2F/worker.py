"""VersionShift v2.1 version-agnostic worker: runs inside ONE frozen scikit-learn environment.

Python >= 3.7, scikit-learn >= 0.21; no walrus/match, no newer-only APIs.
Splits are frozen by the controller; this worker never generates them.

Policies
  D   shipped defaults of this release
  E   archived partial explicit map
  P   STRICT full pin: every reference parameter must transfer unchanged, and the
      target may not add parameters the reference lacks. Anything else fails closed
      with a portability status; nothing is dropped or substituted (v2.0 bug fixed).
  S   SonarSource python:S6973 compliance as written (rule parameters pinned from the pin source; v2.2.5)
  PT  compatibility arm: P plus ONLY the documented translations given in the job
      (status 'semantic_translation'); never pooled with P.

Portability statuses: ok | removed | unsupported | ineffective | unpinned_new | semantic_translation
Job statuses: ok | not_portable | fit_failed
"""
import hashlib, io, json, os, platform, sys, traceback, warnings
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import numpy as np

EXPLICIT = {
    'LR': dict(C=1.0, solver='lbfgs', max_iter=1000, tol=1e-4),
    'SVC': dict(C=1.0, kernel='rbf', gamma='scale'),
    'kNN': dict(n_neighbors=5, weights='uniform', algorithm='brute', metric='euclidean'),
    'RF': dict(n_estimators=300, max_features='sqrt', min_samples_leaf=1, bootstrap=True),
    'HGB': dict(max_iter=200, learning_rate=0.1, early_stopping=False, max_bins=255),
    'GB': dict(n_estimators=200, learning_rate=0.1, max_depth=3, subsample=1.0),
}
CONTROL_PARAMS = ('random_state', 'n_jobs', 'verbose')
# SonarSource rule python:S6973 (hyperparameters that must be set explicitly), for the six study estimators
S6973 = {'LR': (), 'SVC': ('C', 'kernel', 'gamma'), 'kNN': ('n_neighbors',), 'RF': ('min_samples_leaf', 'max_features'),
         'GB': ('learning_rate',), 'HGB': ('learning_rate',)}
PLACEHOLDERS = ('warn', 'deprecated', 'auto_deprecated')
FITTED_ATTRS = ('_gamma', 'n_iter_', 'n_estimators_', 'do_early_stopping_', '_fit_method', 'effective_metric_',
                'max_features_', 'n_features_in_', 'classes_')


class NotPortable(Exception):
    def __init__(self, status, detail):
        Exception.__init__(self, '%s: %s' % (status, detail)); self.status = status; self.detail = detail


def classes():
    try:
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401  (< 1.0 only)
    except Exception:
        pass
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
    return dict(LR=LogisticRegression, SVC=SVC, kNN=KNeighborsClassifier, RF=RandomForestClassifier,
                GB=GradientBoostingClassifier, HGB=HistGradientBoostingClassifier)


def jsonable(v):
    if isinstance(v, (str, bool, int)) or v is None:
        return v
    if isinstance(v, float):
        return v if np.isfinite(v) else repr(v)
    if hasattr(v, 'tolist') and getattr(v, 'size', 11) <= 10:
        return jsonable(v.tolist())
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v][:10]
    return repr(v)[:200]


def defaults_of(cls):
    return {k: jsonable(v) for k, v in cls().get_params(deep=False).items()}


def portable_kwargs(algorithm, pinned, translations, same_release=False):
    """Return (kwargs, report). Raise NotPortable unless every pinned value transfers."""
    cls = classes()[algorithm]
    here = defaults_of(cls)
    report = dict(removed=[], unpinned_new=[], ineffective=[], translated=[])
    kw = {}
    for k, v in pinned.items():
        if k in CONTROL_PARAMS:
            continue
        tr = translations.get(k)
        if tr is not None:
            report['translated'].append(dict(param=k, reference=v, **tr))
            if tr.get('to_param'):
                kw[tr['to_param']] = tr['to_value']
            continue
        if k not in here:
            report['removed'].append(k); continue
        if isinstance(v, str) and v in PLACEHOLDERS and not same_release:
            report['ineffective'].append(k); continue
        kw[k] = v
    for k in here:
        if k not in pinned and k not in CONTROL_PARAMS and k not in translations.get('__new_params_ok__', {}):
            report['unpinned_new'].append(k)
    for status in ('removed', 'ineffective', 'unpinned_new'):
        if report[status]:
            raise NotPortable(status, ','.join(report[status]))
    return kw, report


def construct(algorithm, policy, seed, pinned=None, translations=None, pin_source_sklearn=None):
    cls = classes()[algorithm]
    here = defaults_of(cls)
    report = None
    if policy == 'D':
        kw = {}
    elif policy == 'R':
        # Post-review semantic-restoration condition; not the archived P policy.
        restored = {
            'LR': dict(solver='liblinear', multi_class='ovr'),
            'SVC': dict(gamma='auto'),
            'RF': dict(n_estimators=10),
            'kNN': {},
        }
        if algorithm not in restored:
            raise NotPortable('unsupported', 'G2 restoration not defined for ' + algorithm)
        kw = dict(restored[algorithm])
    elif policy == 'G2E':
        # Post-review diagnostic intervention, NOT the original E or P policy.
        # GB: candidate legacy presort behavior in the new 0.22.2 runtime.
        # HGB: test 255 bins inside the original 0.21.3 runtime.
        import sklearn
        ver = sklearn.__version__
        if algorithm == 'GB' and ver == '0.22.2.post1':
            kw = dict(presort='auto')
        elif algorithm == 'HGB' and ver == '0.21.3':
            kw = dict(max_bins=255)
        else:
            raise NotPortable('unsupported', 'G2E version/estimator mismatch: %s %s' % (algorithm, ver))
    elif policy == 'E':
        kw = dict(EXPLICIT[algorithm])
    elif policy == 'S':
        # SonarSource python:S6973 compliance "as written": only the rule's parameters are explicit, with the
        # values resolved in the pin-source release; everything else stays at this release's shipped default.
        # A value the target release rejects is a hard failure (not_portable/unsupported), never rewritten.
        if pinned is None:
            raise ValueError('policy S requires pinned reference parameters')
        import sklearn
        kw = {}
        for k in S6973.get(algorithm, ()):
            if k not in pinned:
                continue
            if k not in here:
                raise NotPortable('removed', k)
            v = pinned[k]
            if isinstance(v, str) and v in PLACEHOLDERS and pin_source_sklearn != sklearn.__version__:
                raise NotPortable('ineffective', k)
            kw[k] = v
    elif policy in ('P', 'PT'):
        if pinned is None:
            raise ValueError('policy %s requires pinned reference parameters' % policy)
        import sklearn
        kw, report = portable_kwargs(algorithm, pinned, (translations or {}) if policy == 'PT' else {},
                                     same_release=(pin_source_sklearn == sklearn.__version__))
    else:
        raise ValueError('unknown policy')
    if 'random_state' in here and algorithm in ('RF', 'GB', 'HGB'):
        kw['random_state'] = int(seed)
    if 'n_jobs' in here and algorithm in ('RF', 'kNN'):
        kw['n_jobs'] = 1
    try:
        est = cls(**kw)
    except (TypeError, ValueError) as e:
        raise NotPortable('unsupported', repr(e))
    if policy in ('P', 'PT', 'S') and hasattr(est, '_validate_params'):     # sklearn >= 1.2 validates eagerly
        try:
            est._validate_params()
        except (TypeError, ValueError) as e:
            raise NotPortable('unsupported', repr(e))
    resolved = {k: jsonable(v) for k, v in est.get_params(deep=False).items()}
    for k, v in kw.items():
        if resolved.get(k) != jsonable(v):
            raise NotPortable('unsupported', 'constructor did not preserve %s' % k)
    return est, report


def provenance():
    import sklearn, scipy
    out = dict(python=sys.version.split()[0], executable=sys.executable, machine=platform.machine(),
               platform=platform.platform(), processor=platform.processor(), sklearn=sklearn.__version__,
               sklearn_file=sklearn.__file__, numpy=np.__version__, scipy=scipy.__version__)
    try:
        from threadpoolctl import threadpool_info
        out['threadpools'] = [{k: jsonable(d.get(k)) for k in ('internal_api', 'version', 'num_threads', 'filepath', 'architecture')}
                              for d in threadpool_info()]
    except Exception as e:
        out['threadpools'] = 'unavailable: %r' % (e,)
    buf = io.StringIO(); old = sys.stdout
    try:
        sys.stdout = buf; np.show_config()
    except Exception:
        pass
    finally:
        sys.stdout = old
    out['numpy_config_sha256'] = hashlib.sha256(buf.getvalue().encode()).hexdigest()
    return out


def impute_train_median(Xtr, Xte):
    """Version-agnostic, train-fold-only median imputation (numpy only)."""
    if not (np.isnan(Xtr).any() or np.isnan(Xte).any()):
        return Xtr, Xte, 0
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med), 0.0, med)
    n = int(np.isnan(Xtr).sum() + np.isnan(Xte).sum())
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    return Xtr, Xte, n


def fitted_summary(est):
    out = {}
    for a in FITTED_ATTRS:
        if hasattr(est, a):
            try:
                out[a] = jsonable(getattr(est, a))
            except Exception:
                pass
    if hasattr(est, 'coef_'):
        out['coef_shape'] = list(np.shape(est.coef_))
    if hasattr(est, 'estimators_'):
        try:
            out['n_fitted_estimators'] = int(np.size(est.estimators_))
        except Exception:
            pass
    return out


def run(job_path):
    job = json.load(open(job_path))
    out_json = job['out_prefix'] + '.json'; out_npz = job['out_prefix'] + '.npz'
    if os.path.exists(out_json) or os.path.exists(out_npz):
        raise SystemExit('refusing to overwrite ' + out_json)
    rec = dict(id=job['id'], status='fit_failed', provenance=provenance(),
               worker_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(), job_sha256=job.get('job_sha256'))
    arrays = {}
    try:
        if rec['provenance']['sklearn'] != job['expected_sklearn']:
            raise ValueError('runtime mismatch %s != %s' % (rec['provenance']['sklearn'], job['expected_sklearn']))
        data = np.load(job['dataset_npz'], allow_pickle=False)
        X = np.asarray(data['X'], dtype=np.float64); y = np.asarray(data['y'], dtype=np.int64)
        folds = np.load(job['folds_npz'], allow_pickle=False)
        k = int(y.max()) + 1
        scores, conv, imputed, fitted = [], 0, 0, []
        params_seen = report = None
        for key in job['fold_keys']:
            tr = folds[key + '_train']; te = folds[key + '_test']
            est, report = construct(job['algorithm'], job['policy'], job['seeds'][key], job.get('pinned'),
                                    job.get('translations'), job.get('pin_source_sklearn'))
            params_seen = {kk: jsonable(v) for kk, v in est.get_params(deep=False).items()}
            Xtr, Xte, n_imp = impute_train_median(X[tr], X[te]); imputed += n_imp
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                try:
                    est.fit(Xtr, y[tr])
                except ValueError as e:
                    if job['policy'] in ('P', 'PT', 'S') and ('parameter' in str(e).lower() or 'support' in str(e).lower()):
                        raise NotPortable('unsupported', repr(e))
                    raise
                conv += sum('onverg' in str(x.message) for x in w)
            pred = np.asarray(est.predict(Xte), dtype=np.int64)
            if pred.shape != te.shape or pred.min() < 0 or pred.max() >= k:
                raise ValueError('incomplete or invalid prediction array for ' + key)
            arrays[key + '_pred'] = pred
            scores.append(float(np.mean([np.mean(pred[y[te] == c] == c) for c in range(k) if np.any(y[te] == c)])))
            fitted.append(fitted_summary(est))
        tmp = out_npz + '.tmp.npz'
        np.savez_compressed(tmp, **arrays); os.replace(tmp, out_npz)
        rec.update(status='ok', portability='semantic_translation' if report and report['translated'] else 'ok',
                   portability_report=report, fold_keys=job['fold_keys'], ba=scores, constructor=params_seen,
                   fitted_effective=fitted, convergence_warnings=conv, values_imputed=imputed,
                   npz_sha256=hashlib.sha256(open(out_npz, 'rb').read()).hexdigest())
    except NotPortable as e:
        rec.update(status='not_portable', portability=e.status, detail=e.detail)
    except Exception as e:
        rec.update(error=repr(e), traceback=traceback.format_exc(limit=4))
    with open(out_json + '.tmp', 'w') as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    os.replace(out_json + '.tmp', out_json)
    return rec


if __name__ == '__main__':
    r = run(sys.argv[1])
    print(r['id'], r['status'], r.get('portability'))
