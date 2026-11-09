import os
import json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code')); sys.path.insert(0, str(ROOT / 'crossversion'))
import worker as W                         # noqa: E402
import attribution as AT                   # noqa: E402
import estimands_v21 as ES                 # noqa: E402
import calibrate_v21 as CA                 # noqa: E402
PY = sys.executable


class StrictPinTests(unittest.TestCase):
    """Review P0-1: policy P must fail closed instead of dropping parameters."""
    def ref(self):
        return W.defaults_of(W.classes()['RF'])

    def test_same_release_pin_is_portable(self):
        kw, rep = W.portable_kwargs('RF', self.ref(), {}, same_release=True)
        self.assertFalse(any(rep[k] for k in ('removed', 'unpinned_new', 'ineffective')))

    def test_removed_parameter_fails_closed(self):
        pinned = dict(self.ref(), min_impurity_split=None)          # existed in 0.21, removed later
        with self.assertRaises(W.NotPortable) as cm:
            W.portable_kwargs('RF', pinned, {})
        self.assertEqual(cm.exception.status, 'removed')

    def test_new_parameter_in_target_fails_closed(self):
        pinned = self.ref(); pinned.pop('monotonic_cst')            # added after 0.21
        with self.assertRaises(W.NotPortable) as cm:
            W.portable_kwargs('RF', pinned, {})
        self.assertEqual(cm.exception.status, 'unpinned_new')

    def test_placeholder_is_ineffective_across_releases_only(self):
        lr = W.defaults_of(W.classes()['LR'])
        self.assertEqual(lr['penalty'], 'deprecated')
        with self.assertRaises(W.NotPortable) as cm:
            W.portable_kwargs('LR', lr, {}, same_release=False)
        self.assertEqual(cm.exception.status, 'ineffective')
        W.portable_kwargs('LR', lr, {}, same_release=True)

    def test_unsupported_value_fails_closed(self):
        pinned = dict(self.ref(), max_features='auto')              # removed value in >= 1.3
        with self.assertRaises(W.NotPortable) as cm:
            est, _ = W.construct('RF', 'P', 0, pinned, None, pin_source_sklearn='other')
            est.fit(np.random.rand(20, 3), np.arange(20) % 2)
        self.assertIn(cm.exception.status, ('unsupported',))

    def test_documented_translation_only_in_PT(self):
        pinned = dict(self.ref(), min_impurity_split=None)
        tr = {'min_impurity_split': dict(to_param=None, to_value=None, rule='removed in 1.0; None meant disabled', source='changelog')}
        kw, rep = W.portable_kwargs('RF', pinned, tr)
        self.assertEqual(len(rep['translated']), 1)
        with self.assertRaises(W.NotPortable):
            W.construct('RF', 'P', 0, pinned, tr, pin_source_sklearn='other')   # P ignores translations


class AttributionTests(unittest.TestCase):
    """Review P0-2: symmetric 2x2 decomposition with interaction and identifiability."""
    def test_shapley_sums_to_total_and_exposes_interaction(self):
        r = AT.decompose_unit(dict(A=0.0, B=1.0, C=3.0, D=5.0))
        self.assertAlmostEqual(r['defaults'] + r['runtime_stack'], r['total'])
        self.assertAlmostEqual(r['interaction'], 5.0 - 1.0 - 3.0 + 0.0)
        self.assertEqual(r['identifiability'], 'full')

    def test_additive_truth_recovered(self):
        rng = np.random.default_rng(0); rows = []
        for k in range(40):
            for p in range(5):
                a = rng.normal(); dflt, rt = 2.0, 0.5
                for cell, v in dict(A=a, B=a + rt, C=a + dflt, D=a + rt + dflt).items():
                    rows.append(dict(dataset=k, pair=p, cell=cell, theta=v))
        s = AT.summarise(AT.decompose_table(pd.DataFrame(rows)))['full']
        self.assertAlmostEqual(s['defaults']['mean'], 2.0); self.assertAlmostEqual(s['runtime_stack']['mean'], 0.5)
        self.assertAlmostEqual(s['interaction']['mean'], 0.0)

    def test_missing_cells_are_not_given_numbers(self):
        self.assertEqual(AT.decompose_unit(dict(A=0, D=1))['identifiability'], 'total_only')
        self.assertNotIn('defaults', AT.decompose_unit(dict(A=0, D=1)))
        self.assertEqual(AT.decompose_unit(dict(A=0, B=1, D=2))['identifiability'], 'runtime_first')


class EstimandTests(unittest.TestCase):
    def test_pure_null_gives_exact_zero(self):
        S = CA.generate(np.random.default_rng(1), 10, CA.SCENARIOS['N1_pure_null'])
        dl = ES.dataset_level(S)
        self.assertTrue(np.all(dl['M'] == 0)); self.assertTrue(np.all(dl['E'] == 0))
        self.assertTrue(ES.infer(dl)['primary']['below_margin_equivalence'])

    def test_known_shift_recovered(self):
        S = CA.generate(np.random.default_rng(2), 30, CA.SCENARIOS['N1_pure_null'])
        S[:, 1, 3] += 0.03                       # +3 pp on algorithm 3 in the new version, every fold
        dl = ES.dataset_level(S)
        self.assertAlmostEqual(float(dl['M'].mean()), 3.0 * 5 / 15, places=6)

    def test_multiplicity(self):
        np.testing.assert_allclose(ES.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        np.testing.assert_allclose(ES.bh([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])


@unittest.skipIf(os.environ.get('VS_FAST_TESTS') == '1', 'slow end-to-end test (set VS_FAST_TESTS=0 to run)')
class EndToEndTests(unittest.TestCase):
    """Review section 7: harness -> collect -> replay -> v2.1 inferential tables."""
    def test_aa_and_injected_effect_through_final_tables(self):
        ctl = str(ROOT / 'crossversion/controller.py'); an = str(ROOT / 'code/analysis_v21.py')
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            run = lambda *a: subprocess.run([PY, '-I'] + list(a), check=True, capture_output=True, text=True)
            run(ctl, 'freeze', '--plan-dir', str(t / 'plan'), '--reps', '6', '--source', 'bundled')
            run(ctl, 'run', '--plan-dir', str(t / 'plan'), '--out', str(t / 'run'), '--aa', '--release', 'A=' + PY,
                '--release', 'B=' + PY, '--policies', 'D,P', '--pin-from', 'A', '--algorithms', 'LR,kNN,RF')
            run(ctl, 'collect', '--plan-dir', str(t / 'plan'), '--out', str(t / 'run'))
            self.assertIn('REPLAY PASS', run(ctl, 'replay', '--plan-dir', str(t / 'plan'), '--out', str(t / 'run')).stdout)
            run(an, 'crossrelease', '--inp', str(t / 'run/CROSS_RELEASE_FOLDS.csv.gz'), '--edges', 'A:B', '--out', str(t / 'an'))
            rep = json.loads((t / 'an/V21_REPORT.json').read_text())['contrasts'][0]
            self.assertEqual(rep['primary']['estimate_pp'], 0.0); self.assertTrue(rep['primary']['below_margin_equivalence'])
            df = pd.read_csv(t / 'run/CROSS_RELEASE_FOLDS.csv.gz')
            df.loc[(df.release == 'B') & (df.algo == 'RF'), 'ba'] += 0.03
            df.to_csv(t / 'inj.csv.gz', index=False)
            run(an, 'crossrelease', '--inp', str(t / 'inj.csv.gz'), '--edges', 'A:B', '--out', str(t / 'an2'))
            rep = json.loads((t / 'an2/V21_REPORT.json').read_text())['contrasts'][0]
            self.assertAlmostEqual(rep['primary']['estimate_pp'], 3.0 * 2 / 3, places=6)   # RF in 2 of 3 pairs


class OracleTests(unittest.TestCase):
    def test_evaluate_refuses_incomplete_frame(self):
        with tempfile.TemporaryDirectory() as t:
            fr = Path(t) / 'f.csv'
            subprocess.run([PY, '-I', str(ROOT / 'code/oracle_frame.py'), 'build', '--out', str(fr)], check=True, capture_output=True)
            obs = pd.read_csv(fr)[['edge', 'estimator']].assign(observed_change=0); obs.to_csv(Path(t) / 'o.csv', index=False)
            r = subprocess.run([PY, '-I', str(ROOT / 'code/oracle_frame.py'), 'evaluate', '--frame', str(fr), '--observed', str(Path(t) / 'o.csv')],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0); self.assertIn('incomplete', r.stderr + r.stdout)


if __name__ == '__main__':
    unittest.main()
