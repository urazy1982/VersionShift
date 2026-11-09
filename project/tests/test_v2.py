import os
import hashlib, json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import regimes as R                       # noqa: E402
import analyze_emulation as A             # noqa: E402
import run_emulation as E                 # noqa: E402


class RegimeTests(unittest.TestCase):
    def test_negative_controls_identical_across_regimes(self):
        for a in ('kNN', 'GB', 'HGB'):
            for k in (2, 3):
                self.assertEqual(R.effective_signature(a, 'R21', 'D', k), R.effective_signature(a, 'R19', 'D', k))

    def test_documented_default_changes_differ(self):
        for a in ('LR', 'SVC', 'RF'):
            self.assertNotEqual(R.effective_signature(a, 'R21', 'D', 2), R.effective_signature(a, 'R19', 'D', 2))

    def test_explicit_map_neutralises_pinned_changes_but_not_multiclass_lr(self):
        for a in ('SVC', 'RF'):
            self.assertEqual(R.effective_signature(a, 'R21', 'E', 3), R.effective_signature(a, 'R19', 'E', 3))
        self.assertEqual(R.effective_signature('LR', 'R21', 'E', 2), R.effective_signature('LR', 'R19', 'E', 2))
        self.assertNotEqual(R.effective_signature('LR', 'R21', 'E', 3), R.effective_signature('LR', 'R19', 'E', 3))

    def test_r21_rf_only_changes_n_estimators(self):
        a = dict(R.effective_signature('RF', 'R21', 'D', 2)[1]); b = dict(R.effective_signature('RF', 'R19', 'D', 2)[1])
        self.assertEqual({k for k in a if a[k] != b[k]}, {'n_estimators'})


class EmulationTests(unittest.TestCase):
    @unittest.skipIf(os.environ.get('VS_FAST_TESTS') == '1', 'slow end-to-end test (set VS_FAST_TESTS=0 to run)')
    def test_dataset_run_is_reproducible(self):
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        ds = dict(name='iris_t', origin='real', hetero=None, X=X.astype(float), y=y.astype(np.int64))
        r1, d1, _, _ = E.run_dataset((ds, 1)); r2, d2, _, _ = E.run_dataset((ds, 1))
        self.assertEqual([x['pred_sha'] for x in r1], [x['pred_sha'] for x in r2])
        # negative controls share one fit across regimes
        for x in r1:
            if x['algo'] in ('kNN', 'GB', 'HGB'):
                twin = [z for z in r1 if z['algo'] == x['algo'] and z['fold'] == x['fold'] and z['prep'] == x['prep']
                        and z['policy'] == x['policy']]
                self.assertEqual(len({z['pred_sha'] for z in twin}), 1)


class StatisticsTests(unittest.TestCase):
    def test_frozen_posterior_hash(self):
        self.assertEqual(hashlib.sha256((ROOT / 'code/frozen_metrics_v36.py').read_bytes()).hexdigest(), A.FROZEN_SHA)

    def test_corrected_state(self):
        self.assertEqual(A.corrected_state(np.full(30, 0.0)), '~')
        self.assertEqual(A.corrected_state(np.full(30, 0.02)), 'A>B')
        rng = np.random.default_rng(0)
        self.assertEqual(A.corrected_state(rng.normal(0, .05, 30)), '~')

    def test_logratio_statistic_symmetric_under_null(self):
        rng = np.random.default_rng(3); s = []
        for _ in range(20000):
            m = rng.normal(0, 1, 3)            # three iid block means, no version effect
            s.append(np.log((m[1] - m[0]) ** 2) - np.log((m[1] - m[2]) ** 2))
        self.assertLess(abs(np.mean(s)), 0.05)
        self.assertLess(abs(np.mean(np.array(s) > 0) - 0.5), 0.01)


@unittest.skipIf(os.environ.get('VS_FAST_TESTS') == '1', 'slow end-to-end test (set VS_FAST_TESTS=0 to run)')
class CrossVersionAATest(unittest.TestCase):
    def test_aa_bitwise_identical(self):
        ctl = str(ROOT / 'crossversion/controller.py'); py = sys.executable
        with tempfile.TemporaryDirectory() as t:
            plan, out = Path(t) / 'plan', Path(t) / 'run'
            subprocess.run([py, '-I', ctl, 'freeze', '--plan-dir', str(plan), '--reps', '1', '--source', 'bundled'], check=True, capture_output=True)
            subprocess.run([py, '-I', ctl, 'run', '--plan-dir', str(plan), '--out', str(out), '--aa', '--release', 'A=' + py,
                            '--release', 'B=' + py, '--policies', 'D,P', '--pin-from', 'A', '--algorithms', 'LR,RF'],
                           check=True, capture_output=True)
            subprocess.run([py, '-I', ctl, 'collect', '--plan-dir', str(plan), '--out', str(out)], check=True, capture_output=True)
            import pandas as pd
            d = pd.read_csv(out / 'CROSS_RELEASE_FOLDS.csv.gz')
            w = d.pivot_table(index=['dataset', 'algo', 'policy', 'rep', 'fold'], columns='release', values='pred_sha', aggfunc='first')
            self.assertTrue((w.A == w.B).all())
            self.assertTrue((d.status == 'ok').all())


if __name__ == '__main__':
    unittest.main()
