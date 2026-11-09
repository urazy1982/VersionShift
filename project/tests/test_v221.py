"""Regression tests for every finding of the v2.2 independent audit."""
import json, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import decision_v221 as DV                 # noqa: E402
import analysis_v221 as AN                 # noqa: E402
import attribution as AT                   # noqa: E402
PY = sys.executable
ANP = str(ROOT / 'code/analysis_v221.py'); ORP = str(ROOT / 'code/oracle_frame.py')


def run(*a, ok=True):
    r = subprocess.run([PY, '-I'] + list(a), capture_output=True, text=True)
    if ok and r.returncode != 0:
        raise AssertionError(r.stderr[-800:])
    return r


def rows(release, policy, algos, shift=None, datasets=('d1', 'd2', 'd3', 'd4'), not_portable=(), reps=6, folds=5):
    out = []
    for ds in datasets:
        for a in algos:
            if (ds, a) in not_portable:
                out.append(dict(release=release, dataset=ds, algo=a, policy=policy, status='not_portable', portability='removed')); continue
            for r in range(reps):
                for f in range(folds):
                    base = {'LR': .8, 'RF': .85, 'kNN': .75, 'SVC': .82}[a] + 0.002 * ((hash((ds, r, f)) % 7) - 3)
                    out.append(dict(release=release, dataset=ds, algo=a, policy=policy, status='ok', portability='ok',
                                    rep=r, fold=f, ba=base + (shift or {}).get(a, 0.0)))
    return out


class HolmTests(unittest.TestCase):            # audit P0-1
    def test_known_pvalues(self):
        np.testing.assert_allclose(DV.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        np.testing.assert_allclose(DV.bh([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])

    def fake(self, p_mean, p_prev=1.0):
        return dict(q_mean=dict(p_changed_unadjusted=p_mean),
                    q_prev=dict(primary_avg_pair=dict(population='superpopulation', p_exceeds_limit=p_prev, estimate=0.0, upper=0.5),
                                sensitivity_any_pair_pi_max=dict(population='superpopulation', lower=0.0, estimate=0.0)),
                    equivalence_inputs=dict(all_pairs_tost=False, prevalence_ok=False, M_upper_below_margin=False))

    def test_labels_follow_adjusted_not_raw_pvalues(self):
        res = {'e1': self.fake(0.01), 'e2': self.fake(0.04), 'e3': self.fake(0.03), 's1': self.fake(0.04)}
        DV.label_edges(res, ['e1', 'e2', 'e3'])
        self.assertEqual(res['e1']['labels'], ['CHANGED_MEAN'])          # Holm 0.03
        self.assertEqual(res['e2']['labels'], ['INCONCLUSIVE'])          # raw 0.04 < .05 but Holm 0.06
        self.assertEqual(res['e3']['labels'], ['INCONCLUSIVE'])          # Holm 0.06
        self.assertEqual(res['e1']['adjusted']['mean']['method'], 'holm')
        self.assertEqual(res['s1']['adjusted']['mean']['method'], 'bh')  # secondary family adjusted separately
        self.assertEqual(res['s1']['labels'], ['CHANGED_MEAN'])

    def test_one_sided_pvalue_matches_scipy(self):
        from scipy import stats
        x = np.array([0.6, 0.9, 0.7, 1.2, 0.4, 0.8])
        self.assertAlmostEqual(DV.p_mean_exceeds(x, 0.5), stats.ttest_1samp(x, 0.5, alternative='greater').pvalue)

    def test_edge_cli_applies_holm(self):
        algos = ['LR', 'RF']
        df = pd.DataFrame(rows('a', 'D', algos) + rows('b', 'D', algos, {'RF': 0.03}) + rows('c', 'D', algos, {'RF': 0.0}))
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'x.csv'; df.to_csv(p, index=False)
            run(ANP, 'edge', '--inp', str(p), '--edges', 'a:b,a:c', '--primary', 'a:b,a:c', '--unblinded-exploratory', '--out', str(Path(t) / 'e.json'))
            r = json.loads((Path(t) / 'e.json').read_text())['margins']['0.5']
        self.assertEqual(r['a:b']['adjusted']['mean']['method'], 'holm')
        self.assertIn('CHANGED_MEAN', r['a:b']['labels']); self.assertNotIn('CHANGED_MEAN', r['a:c']['labels'])


class StrictFoldTests(unittest.TestCase):       # audit P0-2
    def base(self):
        return pd.DataFrame(rows('old', 'D', ['LR', 'RF']) + rows('new', 'D', ['LR', 'RF']))

    def test_complete_input_passes(self):
        AN.validate(self.base(), 6, 5)

    def test_duplicate_identity_rejected(self):
        df = self.base(); df = pd.concat([df, df.iloc[[5]]])
        with self.assertRaises(AN.InvalidInput) as cm:
            AN.validate(df, 6, 5)
        self.assertIn('duplicate', str(cm.exception))

    def test_missing_fold_rejected(self):
        df = self.base(); df = df.drop(df[(df.release == 'new') & (df.algo == 'RF') & (df.rep == 2) & (df.fold == 3)].index)
        with self.assertRaises(AN.InvalidInput) as cm:
            AN.validate(df, 6, 5)
        self.assertIn('29/30', str(cm.exception))

    def test_unexpected_fold_rejected(self):
        df = self.base(); extra = df.iloc[[0]].copy(); extra['rep'] = 9
        with self.assertRaises(AN.InvalidInput):
            AN.validate(pd.concat([df, extra]), 6, 5)

    def test_reduced_fold_plan_cannot_pass_as_full(self):
        df = pd.DataFrame(rows('old', 'D', ['LR', 'RF'], reps=3) + rows('new', 'D', ['LR', 'RF'], reps=3))
        with self.assertRaises(AN.InvalidInput):
            AN.validate(df, 6, 5)

    def test_pivot_refuses_duplicates_even_if_validate_skipped(self):
        df = self.base(); df = pd.concat([df, df.iloc[[0]]])
        with self.assertRaises(ValueError):
            AN.pair_deltas(df[df.status == 'ok'], 'old', 'D', ['LR', 'RF'])


class OracleCompletenessTests(unittest.TestCase):   # audit P1-1
    def coded(self, t):
        fr = t / 'f.csv'; run(ORP, 'build', '--out', str(fr))
        f = pd.read_csv(fr, keep_default_na=False)
        f['coding_status'] = 'coded_complete'; f['predicted_impact_category'] = 'no_change_expected'
        for c in ('trigger_condition', 'applicable_data_types', 'estimated_compatibility', 'release_note_reference', 'coding_uncertainty'):
            f[c] = 'x'
        f['coder'] = 'UY'; f['coded_on'] = '2026-11-10'
        return fr, f

    def test_exact_key_set_required(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); fr, f = self.coded(t)
            f.to_csv(fr, index=False); run(ORP, 'freeze', '--frame', str(fr), '--out', str(t / 'z.json'))
            pd.concat([f, f.iloc[[0]]]).to_csv(fr, index=False)                       # duplicate key
            self.assertIn('duplicates', run(ORP, 'freeze', '--frame', str(fr), '--out', str(t / 'z.json'), ok=False).stderr)
            f.iloc[1:].to_csv(fr, index=False)                                          # missing key
            self.assertIn('missing', run(ORP, 'freeze', '--frame', str(fr), '--out', str(t / 'z.json'), ok=False).stderr)
            g = f.copy(); g.loc[0, 'estimator'] = 'XGB'; g.to_csv(fr, index=False)     # unexpected key
            self.assertIn('unexpected', run(ORP, 'freeze', '--frame', str(fr), '--out', str(t / 'z.json'), ok=False).stderr)

    def test_provenance_required(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); fr, f = self.coded(t); f.to_csv(fr, index=False)
            Path(str(fr) + '.PROVENANCE.json').unlink()
            self.assertIn('provenance', run(ORP, 'freeze', '--frame', str(fr), '--out', str(t / 'z.json'), ok=False).stderr)


class PopulationTests(unittest.TestCase):          # audit P1-2
    def test_three_population_modes(self):
        a = np.array([True] * 2 + [False] * 18)
        sp = DV.prevalence(a, 'superpopulation'); fc = DV.prevalence(a, 'frame_census')
        st = DV.prevalence(a, 'stratified', weights=np.r_[np.full(10, 3.0), np.full(10, 1.0)])
        self.assertIsNotNone(sp['upper']); self.assertIsNone(fc['upper']); self.assertIn('finite eligible frame', fc['interpretation'])
        self.assertAlmostEqual(st['estimate'], 2 * 3 / 40)                              # both affected in the heavy stratum
        self.assertLess(st['n_effective'], 20)

    def test_stratified_requires_selection(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'x.csv'; pd.DataFrame(rows('o', 'D', ['LR', 'RF']) + rows('n', 'D', ['LR', 'RF'])).to_csv(p, index=False)
            r = run(ANP, 'edge', '--inp', str(p), '--edges', 'o:n', '--population', 'stratified', '--unblinded-exploratory',
                    '--out', str(Path(t) / 'e.json'), ok=False)
            self.assertIn('requires --selection', r.stderr)


class MissingnessTests(unittest.TestCase):         # audit P1-3
    def test_complete_case_primary_and_sensitivity(self):
        D = np.full((10, 3), 0.2); D[:4, 2] = np.nan; D[:4, :2] = 3.0              # missing pair only where shifts are big
        r = DV.decide(D)
        self.assertEqual(r['K_complete_case'], 6); self.assertAlmostEqual(r['complete_case_coverage'], 0.6)
        self.assertAlmostEqual(r['q_mean']['M_pp'], 0.2)                             # complete-case primary
        self.assertGreater(r['missingness_sensitivity']['available_case_M_pp'], 1.0)  # available-case differs: flagged
        self.assertEqual(r['pair_availability'], {0: 10, 1: 10, 2: 6})
        self.assertAlmostEqual(r['missingness_sensitivity']['prevalence_bounds_missing_as_affected'], 0.4)


class OneStrongPairTests(unittest.TestCase):       # audit section 2
    def test_one_changed_pair_is_caught_by_pi_max_not_by_average(self):
        D = np.zeros((40, 15)); D[:, 0] = 4.0
        r = DV.label_edges({'e': DV.decide(D)}, ['e'])['e']
        self.assertEqual(r['q_prev']['primary_avg_pair']['affected'], 0)
        self.assertEqual(r['q_prev']['sensitivity_any_pair_pi_max']['affected'], 40)
        self.assertIn('ANY_PAIR_WIDESPREAD', r['sensitivity_flags'])
        self.assertNotIn('EQUIVALENT_BALANCED', r['labels']); self.assertNotIn('UNIFORMLY_SMALL', r['labels'])

    def test_pi_max_threshold_is_one_pp(self):
        D = np.zeros((40, 15)); D[:, 0] = 0.8                                          # below 1 pp: not flagged
        r = DV.label_edges({'e': DV.decide(D)}, ['e'])['e']
        self.assertEqual(r['q_prev']['sensitivity_any_pair_pi_max']['threshold_pp'], 1.0)
        self.assertEqual(r['sensitivity_flags'], []); self.assertEqual(r['q_prev']['descriptive_pi_max_at_margin'], 1.0)


class SizingTests(unittest.TestCase):              # audit P1-4
    def test_positive_edge_can_drive_K(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            def edge_file(name, M_k):
                D = np.tile(np.asarray(M_k)[:, None], (1, 15))
                r = DV.label_edges({name: DV.decide(D)}, [name]); r[name]['edge'] = name
                (t / (name + '.json')).write_text(json.dumps(dict(margins={'0.5': r}), default=float))
                return str(t / (name + '.json'))
            neg = edge_file('neg', np.abs(np.random.default_rng(0).normal(0.1, 0.05, 16)))
            pos = edge_file('pos', np.abs(np.random.default_rng(1).normal(0.9, 1.2, 16)))
            r = run(ANP, 'size', '--negative', neg, '--positive', pos, '--eligible', '72', '--stage-b-used', '16', '--out', str(t / 's.json'))
            s = json.loads((t / 's.json').read_text())
        req = s['requirements']
        self.assertIn('positive_power90:pos', req)
        self.assertGreater(req['positive_power90:pos'], req['negative_precision:neg'])
        self.assertEqual(s['K_required'], max(v for v in req.values() if isinstance(v, int)))


class MitigationPopulationTests(unittest.TestCase):   # audit section 3
    def test_three_populations_never_pooled(self):
        algos = ['LR', 'RF']; ds = ('d1', 'd2', 'd3', 'd4', 'd5', 'd6')
        df = pd.DataFrame(rows('old', 'D', algos, datasets=ds) + rows('new', 'D', algos, {'RF': 0.04}, datasets=ds) +
                          rows('new', 'P', algos, datasets=ds, not_portable=(('d3', 'RF'), ('d4', 'RF'), ('d5', 'RF'), ('d6', 'RF'))) +
                          rows('new', 'PT', algos, {'RF': 0.01}, datasets=ds, not_portable=(('d5', 'RF'), ('d6', 'RF'))))
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'x.csv'; df.to_csv(p, index=False)
            run(ANP, 'mitigation', '--inp', str(p), '--edge', 'old:new', '--unblinded-exploratory', '--out', str(Path(t) / 'm.json'))
            m = json.loads((Path(t) / 'm.json').read_text())
        self.assertEqual(m['failure_to_transfer']['totals'], {'strictly_portable': 8, 'translation_required': 2, 'non_transferable': 2})
        self.assertEqual(m['mitigation_P']['datasets'], 2); self.assertEqual(m['mitigation_PT']['datasets'], 2)
        self.assertAlmostEqual(m['mitigation_P']['M_pinned'], 0.0, places=9)
        self.assertAlmostEqual(m['mitigation_PT']['M_pinned'], 1.0, places=6)          # translation leaves +1 pp
        self.assertAlmostEqual(m['mitigation_PT']['reduction_pp'], 3.0, places=6)

    def test_runtime_component_is_labelled_runtime_stack(self):
        r = AT.decompose_unit(dict(A=0, B=1, C=2, D=3))
        self.assertIn('runtime_stack', r); self.assertNotIn('runtime', r)


if __name__ == '__main__':
    unittest.main()
