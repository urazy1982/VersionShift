import json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import decision_v22 as DV                  # noqa: E402
import calibrate_v21 as CA                 # noqa: E402
import estimands_v21 as ES                 # noqa: E402
PY = sys.executable
AN = str(ROOT / 'code/analysis_v22.py'); OR = str(ROOT / 'code/oracle_frame.py')


def run(*a, ok=True):
    r = subprocess.run([PY, '-I'] + list(a), capture_output=True, text=True)
    if ok and r.returncode != 0:
        raise AssertionError(r.stderr[-800:])
    return r


class DecisionRuleTests(unittest.TestCase):
    def test_zero_event_prevalence_bound_needs_29_datasets(self):
        self.assertEqual(DV.min_k_zero_events(), 29)

    def test_pure_null_is_uniformly_small_only_with_enough_datasets(self):
        for K, expected in ((16, 'INCONCLUSIVE'), (29, 'UNIFORMLY_SMALL')):
            S = CA.generate(np.random.default_rng(K), K, CA.SCENARIOS['N1_pure_null'])
            self.assertEqual(DV.decide(ES.dataset_level(S)['pair_shift'])['labels'], [expected])

    def test_orientation_invariance(self):
        rng = np.random.default_rng(3); D = rng.normal(0, 0.2, (40, 15)); D2 = D.copy(); D2[:, [1, 4, 9]] *= -1
        a, b = DV.decide(D), DV.decide(D2)
        self.assertEqual(a['labels'], b['labels']); self.assertAlmostEqual(a['q_mean']['M_pp'], b['q_mean']['M_pp'])

    def test_sparse_large_effect_is_not_called_equivalent(self):
        D = np.zeros((60, 15)); D[:12, :5] = 4.0            # 20% of datasets, +4 pp on five pairs
        labels = DV.decide(D)['labels']
        self.assertNotIn('EQUIVALENT_BALANCED', labels); self.assertNotIn('UNIFORMLY_SMALL', labels)
        self.assertIn('WIDESPREAD', labels)

    def test_balanced_but_not_uniformly_small(self):
        rng = np.random.default_rng(5); D = rng.choice([-0.6, 0.6], (200, 15)) * (rng.random((200, 15)) < 0.12)
        r = DV.decide(D)
        self.assertTrue(r['q_mean']['pairs_all_tost_equivalent'])

    def test_missing_pairs_are_nan_not_zero(self):
        D = np.full((30, 3), 2.0); D[:, 2] = np.nan
        r = DV.decide(D); self.assertEqual(r['q_mean']['per_pair'][2]['n'], 0); self.assertAlmostEqual(r['q_mean']['M_pp'], 2.0)


def long_rows(release, policy, algos, shift=None, datasets=('d1', 'd2', 'd3', 'd4'), not_portable=()):
    rng = np.random.default_rng(abs(hash((release, policy))) % 2**32); rows = []
    for ds in datasets:
        for a in algos:
            if (ds, a) in not_portable:
                rows.append(dict(release=release, condition=release, dataset=ds, algo=a, policy=policy, status='not_portable', portability='removed')); continue
            for r in range(6):
                for f in range(5):
                    base = {'LR': .8, 'RF': .85, 'kNN': .75}[a] + 0.01 * np.sin(hash((ds, r, f)) % 97)
                    rows.append(dict(release=release, condition=release, dataset=ds, algo=a, policy=policy, status='ok', portability='ok',
                                     rep=r, fold=f, ba=base + (shift or {}).get(a, 0.0)))
    return rows


class SelectiveMissingnessTests(unittest.TestCase):
    """Amendment A-SM: mitigation is estimated only on portable units; coverage and failure-to-transfer are outcomes."""
    def test_conditional_mitigation_and_coverage(self):
        algos = ['LR', 'kNN', 'RF']
        rows = (long_rows('old', 'D', algos) + long_rows('new', 'D', algos, shift={'RF': 0.04}) +
                long_rows('new', 'P', algos, not_portable=(('d3', 'RF'), ('d4', 'RF'))))
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'x.csv'; pd.DataFrame(rows).to_csv(p, index=False)
            run(AN, 'mitigation', '--inp', str(p), '--edge', 'old:new', '--unblinded-exploratory', '--out', str(Path(t) / 'm.json'))
            m = json.loads((Path(t) / 'm.json').read_text())
        rf = [x for x in m['failure_to_transfer']['by_estimator'] if x['algo'] == 'RF'][0]
        self.assertEqual((rf['planned'], rf['portable'], rf['not_portable']), (4, 2, 2))
        self.assertEqual(m['compatibility_coverage']['compatible_pair_units'], 4 * 3 - 2 * 2)
        mc = m['mitigation_conditional_on_compatibility']
        self.assertAlmostEqual(mc['M_pinned'], 0.0, places=9)
        self.assertGreater(mc['reduction_pp'], 0)
        self.assertIn('portable configurations', mc['scope'])


class GateAndAttributionTests(unittest.TestCase):
    def test_gate_refuses_without_frozen_oracle(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'x.csv'; pd.DataFrame(long_rows('old', 'D', ['LR', 'RF']) + long_rows('new', 'D', ['LR', 'RF'])).to_csv(p, index=False)
            r = run(AN, 'edge', '--inp', str(p), '--edges', 'old:new', '--out', str(Path(t) / 'e.json'), ok=False)
            self.assertNotEqual(r.returncode, 0); self.assertIn('blinding gate', r.stderr + r.stdout)
            run(AN, 'edge', '--inp', str(p), '--edges', 'old:new', '--unblinded-exploratory', '--out', str(Path(t) / 'e.json'))
            self.assertEqual(json.loads((Path(t) / 'e.json').read_text())['header']['blinding'], 'UNBLINDED_EXPLORATORY')

    def test_attribution_cli_recovers_default_and_runtime_parts(self):
        algos = ['LR', 'RF']
        rows = (long_rows('old', 'D', algos) + long_rows('new', 'Pold', algos, shift={'RF': 0.01}) +
                long_rows('old', 'Pnew', algos, shift={'RF': 0.03}) + long_rows('new', 'D', algos, shift={'RF': 0.04}))
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'x.csv'; pd.DataFrame(rows).to_csv(p, index=False)
            run(AN, 'attribution', '--unblinded-exploratory', '--cell', 'A=%s@old:D' % p, '--cell', 'B=%s@new:Pold' % p,
                '--cell', 'C=%s@old:Pnew' % p, '--cell', 'D=%s@new:D' % p, '--out', str(Path(t) / 'a.json'))
            s = json.loads((Path(t) / 'a.json').read_text())['summary']['full']
        self.assertAlmostEqual(s['total']['mean'], -4.0, places=6)           # pair LR-RF: RF +4 pp -> LR-RF -4 pp
        self.assertAlmostEqual(s['runtime_stack']['mean'], -1.0, places=6)
        self.assertAlmostEqual(s['defaults']['mean'], -3.0, places=6)
        self.assertAlmostEqual(s['interaction']['mean'], 0.0, places=6)


class OracleProtocolTests(unittest.TestCase):
    def test_freeze_refuses_draft_and_incomplete_coding(self):
        with tempfile.TemporaryDirectory() as t:
            fr = Path(t) / 'f.csv'; run(OR, 'build', '--out', str(fr))
            r = run(OR, 'freeze', '--frame', str(fr), '--out', str(Path(t) / 'z.json'), ok=False)
            self.assertNotEqual(r.returncode, 0); self.assertIn('78 of 78', r.stderr + r.stdout)

    def test_recode_window_and_kappa(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); fr = t / 'f.csv'; run(OR, 'build', '--out', str(fr))
            f = pd.read_csv(fr, keep_default_na=False)
            cats = ['no_change_expected', 'change_expected', 'change_possible', 'fp_level_change_possible']
            f['coding_status'] = 'coded_complete'; f['predicted_impact_category'] = [cats[i % 4] for i in range(len(f))]
            for c in ('trigger_condition', 'applicable_data_types', 'estimated_compatibility', 'release_note_reference', 'coding_uncertainty'):
                f[c] = f[c].replace('', 'x')
            f['coder'] = 'UY'; f['coded_on'] = '2026-11-10'; f.to_csv(fr, index=False)
            run(OR, 'freeze', '--frame', str(fr), '--out', str(t / 'z.json'))
            run(OR, 'recode-sample', '--frame', str(fr), '--frozen', str(t / 'z.json'), '--out', str(t / 'r.csv'))
            r = pd.read_csv(t / 'r.csv', keep_default_na=False); self.assertEqual(len(r), 20)
            self.assertTrue((r.predicted_impact_category == '').all())          # labels concealed
            orig = f.set_index(['edge', 'estimator']).predicted_impact_category
            r['predicted_impact_category'] = [orig[(e, s)] for e, s in zip(r.edge, r.estimator)]
            for i in r.index[:2]:
                r.loc[i, 'predicted_impact_category'] = [c for c in cats if c != r.loc[i, 'predicted_impact_category']][0]
            r['coded_on'] = '2026-11-01'; r.to_csv(t / 'r.csv', index=False)
            bad = run(OR, 'agreement', '--frame', str(fr), '--recode', str(t / 'r.csv'), '--out-adjudication', str(t / 'adj.csv'), ok=False)
            self.assertIn('14-day', bad.stderr + bad.stdout)
            key = json.loads((t / 'r.csv.KEY.json').read_text()); r['coded_on'] = key['earliest_recode']; r.to_csv(t / 'r.csv', index=False)
            out = run(OR, 'agreement', '--frame', str(fr), '--recode', str(t / 'r.csv'), '--out-adjudication', str(t / 'adj.csv')).stdout
            res = json.loads(out); self.assertEqual(res['n'], 20); self.assertLess(res['cohen_kappa'], 1.0)
            self.assertEqual(res['kind'], 'intra-rater (same author)')


if __name__ == '__main__':
    unittest.main()
