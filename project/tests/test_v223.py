"""v2.2.3 regression tests: plan coverage for mitigation and attribution; G2-consistent handling of failed D jobs."""
import json, subprocess, sys, tempfile, unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import analysis_v221 as A            # noqa: E402
PY = sys.executable; ANP = str(ROOT / 'code/analysis_v221.py')
BASE = dict(LR=.80, SVC=.82, kNN=.75, RF=.85, GB=.84, HGB=.86)


def rows(release, policy, datasets, shift=None, status=None):
    out = []
    for ds in datasets:
        for a in A.ALGOS:
            st = (status or {}).get((ds, a))
            if st == 'absent':
                continue
            if st:
                out.append(dict(release=release, dataset=ds, algo=a, policy=policy, status=st, portability='removed' if st == 'not_portable' else None))
                continue
            for r in range(6):
                for f in range(5):
                    out.append(dict(release=release, dataset=ds, algo=a, policy=policy, status='ok', portability='ok', rep=r, fold=f,
                                    ba=BASE[a] + 0.001 * ((r * 5 + f) % 4) + (shift or {}).get(a, 0.0)))
    return out


def plan(t, datasets):
    d = t / 'plan'; d.mkdir(exist_ok=True)
    (d / 'PLAN.json').write_text(json.dumps(dict(datasets=[dict(name=x) for x in datasets])))
    return str(d)


def run(*a):
    return subprocess.run([PY, '-I', ANP] + list(a), capture_output=True, text=True)


class MitigationCoverageTests(unittest.TestCase):
    DS = ('d1', 'd2', 'd3')

    def frame(self, p_status=None, pt=True):
        r = rows('old', 'D', self.DS) + rows('new', 'D', self.DS, {'RF': 0.04}) + rows('new', 'P', self.DS, status=p_status)
        if pt:
            r += rows('new', 'PT', self.DS, status=p_status)
        return pd.DataFrame(r)

    def test_absent_pinned_job_is_refused_not_dropped(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); p = t / 'x.csv'; self.frame({('d2', 'RF'): 'absent'}).to_csv(p, index=False)
            r = run('mitigation', '--inp', str(p), '--edge', 'old:new', '--plan-dir', plan(t, self.DS), '--unblinded-exploratory', '--out', str(t / 'm.json'))
            self.assertNotEqual(r.returncode, 0); self.assertIn('entirely absent', r.stderr)

    def test_not_portable_record_is_an_outcome(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); p = t / 'x.csv'; self.frame({('d2', 'RF'): 'not_portable'}).to_csv(p, index=False)
            r = run('mitigation', '--inp', str(p), '--edge', 'old:new', '--plan-dir', plan(t, self.DS), '--unblinded-exploratory', '--out', str(t / 'm.json'))
            self.assertEqual(r.returncode, 0, r.stderr)
            m = json.loads((t / 'm.json').read_text())
            self.assertEqual(m['failure_to_transfer']['totals'].get('non_transferable'), 1)
            self.assertEqual(sum(m['failure_to_transfer']['totals'].values()), len(self.DS) * len(A.ALGOS))

    def test_pt_arm_required_unless_declared_absent(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); p = t / 'x.csv'; self.frame(pt=False).to_csv(p, index=False)
            r = run('mitigation', '--inp', str(p), '--edge', 'old:new', '--plan-dir', plan(t, self.DS), '--unblinded-exploratory', '--out', str(t / 'm.json'))
            self.assertNotEqual(r.returncode, 0)
            r = run('mitigation', '--inp', str(p), '--edge', 'old:new', '--plan-dir', plan(t, self.DS), '--no-pt', '--unblinded-exploratory', '--out', str(t / 'm.json'))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotIn('mitigation_PT', json.loads((t / 'm.json').read_text()))


class AttributionCoverageTests(unittest.TestCase):
    def test_absent_cell_job_is_refused(self):
        DS = ('d1', 'd2')
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); p = t / 'x.csv'
            pd.DataFrame(rows('old', 'D', DS) + rows('new', 'D', DS) + rows('new', 'P', DS, status={('d1', 'GB'): 'absent'}) +
                         rows('old', 'P', DS)).to_csv(p, index=False)
            cells = ['A=%s@old:D' % p, 'B=%s@new:P' % p, 'C=%s@old:P' % p, 'D=%s@new:D' % p]
            args = ['attribution', '--plan-dir', plan(t, DS), '--unblinded-exploratory', '--out', str(t / 'a.json')]
            for c in cells:
                args += ['--cell', c]
            r = run(*args)
            self.assertNotEqual(r.returncode, 0); self.assertIn('entirely absent', r.stderr)


class G2FailureLedgerTests(unittest.TestCase):
    def build(self, t, n_ds, failed):
        DS = tuple('d%02d' % i for i in range(n_ds))
        st = {(d, 'SVC'): 'fit_failed' for d in DS[:failed]}
        df = pd.DataFrame(rows('old', 'D', DS) + rows('new', 'D', DS, {'RF': 0.02}, status=st))
        p = t / 'x.csv'; df.to_csv(p, index=False)
        return p, plan(t, DS)

    def test_failed_default_job_within_g2_excludes_dataset_and_is_ledgered(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); p, pl = self.build(t, 20, 1)          # 1/20 = 5%: allowed
            r = run('edge', '--inp', str(p), '--edges', 'old:new', '--primary', 'old:new', '--plan-dir', pl, '--unblinded-exploratory', '--out', str(t / 'e.json'))
            self.assertEqual(r.returncode, 0, r.stderr)
            e = json.loads((t / 'e.json').read_text())
            self.assertEqual(e['plan_coverage']['excluded_datasets_failed_default_jobs'], ['d00'])
            self.assertEqual(e['plan_coverage']['failure_ledger'][0]['status'], 'fit_failed')
            self.assertNotIn('d00', e['margins']['0.5']['old:new']['datasets'])

    def test_failures_beyond_g2_refuse_the_analysis(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); p, pl = self.build(t, 20, 13)         # 13/240 D jobs = 5.42% > 5%
            r = run('edge', '--inp', str(p), '--edges', 'old:new', '--plan-dir', pl, '--unblinded-exploratory', '--out', str(t / 'e.json'))
            self.assertNotEqual(r.returncode, 0); self.assertIn('G2 failed', r.stderr)


if __name__ == '__main__':
    unittest.main()
