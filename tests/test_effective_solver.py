import importlib
import warnings
import numpy as np
import pytest
from sklearn.svm import LinearSVC
from versionshift_v3.worker import fit_with_solver_observation

@pytest.mark.parametrize('dual,n,p',[(True,60,5),(False,60,5),('auto',60,5),('auto',20,30)])
def test_backend_observation_matches_unchanged_fit(dual,n,p):
    rng=np.random.RandomState(42);X=rng.randn(n,p);y=np.arange(n)%2
    module=importlib.import_module(LinearSVC.__module__);original=module._fit_liblinear
    a=LinearSVC(dual=dual,random_state=8,max_iter=5000);b=LinearSVC(dual=dual,random_state=8,max_iter=5000);rows=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');fit_with_solver_observation(a,X,y,'r0_f0',rows);b.fit(X,y)
    assert module._fit_liblinear is original
    assert len(rows)==1 and rows[0]['fit_returned']
    expected=(n<p) if dual=='auto' else dual
    assert rows[0]['effective_dual']==expected
    assert np.array_equal(a.predict(X),b.predict(X)) and np.array_equal(a.coef_,b.coef_)

def test_backend_is_restored_after_failure(monkeypatch):
    module=importlib.import_module(LinearSVC.__module__);original=module._fit_liblinear
    def failing(*args,**kwargs):raise RuntimeError('synthetic backend failure')
    failing.__signature__=__import__('inspect').signature(original)
    monkeypatch.setattr(module,'_fit_liblinear',failing)
    with pytest.raises(RuntimeError,match='synthetic backend failure'):
        fit_with_solver_observation(LinearSVC(dual=True),np.ones((10,2)),np.arange(10)%2,'r0_f0',[])
    assert module._fit_liblinear is failing
