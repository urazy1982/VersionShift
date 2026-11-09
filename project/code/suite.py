"""Offline dataset suite: 4 bundled real datasets + 40 independently generated synthetic families.

Each synthetic family has its own generator seed, so families are independent units.
Half the synthetic families get heterogeneous raw feature scales (10**U(-1.5, 1.5)),
because default changes such as SVC gamma 'auto'->'scale' only bite when feature
variance differs from 1.
"""
import hashlib
import numpy as np
from sklearn import datasets as skd


def h(*parts):
    raw = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(raw[:4], 'big') % (2**31 - 1)


def _synthetic(spec):
    rng = np.random.default_rng(spec['seed'])
    kind = spec['kind']
    n, p = spec['n'], spec['p']
    if kind == 'linear':
        X, y = skd.make_classification(n_samples=n, n_features=p, n_informative=max(2, int(p * spec['inf'])),
                                       n_redundant=max(0, int(p * 0.2)), n_classes=spec['k'],
                                       n_clusters_per_class=spec['clusters'], class_sep=spec['sep'],
                                       flip_y=spec['flip'], weights=spec['weights'], random_state=spec['seed'])
    elif kind == 'moons':
        X0, y = skd.make_moons(n_samples=n, noise=spec['noise'], random_state=spec['seed'])
        X = np.hstack([X0, rng.normal(size=(n, p - 2))])
    elif kind == 'circles':
        X0, y = skd.make_circles(n_samples=n, noise=spec['noise'], factor=0.5, random_state=spec['seed'])
        X = np.hstack([X0, rng.normal(size=(n, p - 2))])
    elif kind == 'xor':
        X = rng.uniform(-1, 1, size=(n, p))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        flip = rng.random(n) < spec['flip']
        y = np.where(flip, 1 - y, y)
    else:
        raise ValueError(kind)
    X = np.asarray(X, dtype=np.float64)
    if spec['hetero']:
        X = X * (10.0 ** rng.uniform(-1.5, 1.5, size=X.shape[1])) + rng.normal(0, 5, size=X.shape[1])
    _, y = np.unique(y, return_inverse=True)
    return X, y.astype(np.int64)


def synthetic_specs(count=40, master=20261106):
    rng = np.random.default_rng(master)
    specs = []
    kinds = ['linear'] * 24 + ['moons'] * 6 + ['circles'] * 4 + ['xor'] * 6
    rng.shuffle(kinds)
    for i, kind in enumerate(kinds[:count]):
        k = int(rng.choice([2, 2, 3, 4])) if kind == 'linear' else 2
        imb = float(rng.choice([0.5, 0.5, 0.7, 0.85]))
        weights = None if k > 2 or imb == 0.5 else [imb, 1 - imb]
        specs.append(dict(name=f'syn{i:02d}_{kind}', kind=kind, seed=h('family', master, i),
                          n=int(rng.choice([300, 600, 1000, 1500])), p=int(rng.choice([6, 10, 20, 30])),
                          k=k, inf=float(rng.choice([0.3, 0.5, 0.8])), clusters=int(rng.choice([1, 2])),
                          sep=float(rng.choice([0.6, 1.0, 1.5])), flip=float(rng.choice([0.0, 0.03, 0.08])),
                          noise=float(rng.choice([0.15, 0.3])), weights=weights, hetero=bool(i % 2)))
    return specs


def load_all():
    out = []
    for name, loader in [('iris', skd.load_iris), ('wine', skd.load_wine),
                         ('breast_cancer', skd.load_breast_cancer), ('digits', skd.load_digits)]:
        X, y = loader(return_X_y=True)
        out.append(dict(name=name, origin='real', X=np.asarray(X, float), y=np.asarray(y, np.int64), hetero=None))
    for spec in synthetic_specs():
        X, y = _synthetic(spec)
        out.append(dict(name=spec['name'], origin='synthetic', X=X, y=y, hetero=spec['hetero'], spec=spec))
    return out
