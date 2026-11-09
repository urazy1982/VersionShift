"""v2.2.5 tests: secondary validation lenses, S6973 policy arm, mitigation experiment."""
import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code')); sys.path.insert(0, str(ROOT / 'crossversion'))
import lenses_v225 as L                     # noqa: E402
import worker as W                          # noqa: E402
PY = sys.executable


class LensBTests(unittest.TestCase):
    def test_reml_recovers_known_heterogeneity(self):
        rng = np.random.default_rng(0); K = 3000; v = np.full(K, 0.25)
        y = rng.normal(1.0, 0.5, K) + rng.normal(0, 0.5, K)
        r = L.random_effects(y, v)
        self.assertAlmostEqual(r['mu'], 1.0, delta=0.05); self.assertAlmostEqual(r['tau'], 0.5, delta=0.05)
        new = rng.normal(1.0, 0.5, 50000)
        self.assertAlmostEqual(np.mean((new >= r['pi'][0]) & (new <= r['pi'][1])), 0.95, delta=0.01)

    def test_no_heterogeneity_gives_tau_zero(self):
        rng = np.random.default_rng(1); y = rng.normal(0, 0.3, 400)
        self.assertLess(L.random_effects(y, np.full(400, 0.09))['tau'], 0.05)


class LensCTests(unittest.TestCase):
    def test_exact_zero_pairs_do_not_mask_real_changes(self):
        """Regression: p = 1 for deterministic unchanged pairs made the Cauchy statistic diverge to -inf."""
        X = np.zeros((1, 15, 6)); rng = np.random.default_rng(2)
        X[0, :5] = 4.0 + rng.normal(0, 0.3, (5, 6))       # five pairs of one algorithm changed
        p = L.rerun_t_pvalues(X)
        self.assertLess(L.combine_dataset(p[0], (np.abs(X[0]) < 1e-12).all(1)), 1e-4)
        self.assertGreater(L.cauchy_combine(p[0]), 0.5)  # the naive combination is masked (documents the bug)

    def test_all_unchanged_dataset_has_p_one(self):
        self.assertEqual(L.lens_c(np.zeros((3, 15, 6)))['discoveries'], 0)

    def test_cauchy_valid_on_independent_nulls(self):
        p = np.random.default_rng(3).random((40000, 15))
        self.assertAlmostEqual(np.mean(L.cauchy_combine(p, 1) < 0.05), 0.05, delta=0.005)

    def test_storey_pi1(self):
        rng = np.random.default_rng(4)
        p = np.r_[rng.random(800), rng.random(200) * 1e-4]
        self.assertAlmostEqual(L.storey_pi1(p), 0.2, delta=0.06)


class LensDTests(unittest.TestCase):
    def test_size_at_the_margin_for_skewed_nonnegative_data(self):
        rng = np.random.default_rng(5)
        rej = np.mean([L.lens_d(rng.exponential(0.5, 300), 0.5, order_seed=i)['rejected'] for i in range(1500)])
        self.assertLessEqual(rej, 0.05)

    def test_power_and_stopping(self):
        rng = np.random.default_rng(6)
        r = L.lens_d(rng.exponential(2.0, 60), 0.5)
        self.assertTrue(r['rejected']); self.assertLess(r['stopping_K'], 60)

    def test_rejects_negative_input(self):
        with self.assertRaises(ValueError):
            L.lens_d(np.array([-0.1, 1.0]), 0.5)


class S6973PolicyTests(unittest.TestCase):
    def test_only_rule_parameters_are_pinned(self):
        ref = W.defaults_of(W.classes()['RF']); ref['n_estimators'] = 10; ref['max_features'] = 'sqrt'
        est, _ = W.construct('RF', 'S', 0, ref, None, pin_source_sklearn='other')
        self.assertEqual(est.get_params()['n_estimators'], W.defaults_of(W.classes()['RF'])['n_estimators'])   # not covered
        self.assertEqual(est.get_params()['max_features'], 'sqrt')

    def test_rule_value_rejected_by_target_is_a_hard_failure(self):
        ref = W.defaults_of(W.classes()['RF']); ref['max_features'] = 'auto'              # 0.21-era value
        with self.assertRaises(W.NotPortable) as cm:
            W.construct('RF', 'S', 0, ref, None, pin_source_sklearn='other')
        self.assertEqual(cm.exception.status, 'unsupported')

    def test_logistic_regression_is_not_covered(self):
        ref = dict(W.defaults_of(W.classes()['LR']), solver='liblinear')
        est, _ = W.construct('LR', 'S', 0, ref, None, pin_source_sklearn='other')
        self.assertEqual(est.get_params()['solver'], W.defaults_of(W.classes()['LR'])['solver'])


class MitigationExperimentTests(unittest.TestCase):
    def test_policy_ordering_on_pilot_data(self):
        res = ROOT / 'results/EXP_EMIT_V225.json'
        if not res.is_file():
            self.skipTest('run code/policy_mitigation_v225.py first')
        r = json.loads(res.read_text())['conditions']
        for prep in ('raw', 'std'):
            M = {k: v['M_pp'] for k, v in r[prep].items()}
            self.assertGreaterEqual(M['D'], M['S6973fix']); self.assertGreater(M['S6973fix'], M['E']); self.assertEqual(M['P'], 0.0)
        self.assertAlmostEqual(r['std']['D']['M_pp'], r['std']['S6973fix']['M_pp'])      # no protection on standardised data


if __name__ == '__main__':
    unittest.main()
