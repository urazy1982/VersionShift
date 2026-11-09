"""G2's 95%-of-jobs threshold and dataset retention are distinct quantities."""
import json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v223 import G2FailureLedgerTests, run


class G2JobLevelTests(unittest.TestCase):
    def test_one_svc_failure_in_16_dataset_stage_b_is_allowed_but_dataset_excluded(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            csv, pl = G2FailureLedgerTests().build(root, 16, 1)
            out = root / 'edge.json'
            r = run('edge', '--inp', str(csv), '--edges', 'old:new', '--primary', 'old:new',
                    '--plan-dir', pl, '--unblinded-exploratory', '--out', str(out))
            self.assertEqual(r.returncode, 0, r.stderr)
            result = json.loads(out.read_text())
            coverage = result['plan_coverage']
            self.assertEqual(coverage['planned_default_jobs'], 16*2*6)
            self.assertEqual(coverage['failed_default_jobs'], 1)
            self.assertAlmostEqual(coverage['default_job_success_share'], 191/192)
            self.assertEqual(coverage['retained_datasets'], 15)
            self.assertEqual(coverage['excluded_datasets_failed_default_jobs'], ['d00'])
            self.assertNotIn('d00', result['margins']['0.5']['old:new']['datasets'])

    def test_threshold_is_inclusive_at_exactly_five_percent(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            csv, pl = G2FailureLedgerTests().build(root, 20, 12)
            r = run('edge', '--inp', str(csv), '--edges', 'old:new', '--plan-dir', pl,
                    '--unblinded-exploratory', '--out', str(root/'edge.json'))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads((root/'edge.json').read_text())['plan_coverage']['failed_default_jobs'], 12)

    def test_one_svc_failure_in_16_dataset_stage_b_does_not_change_missing_job_rule(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            csv, pl = G2FailureLedgerTests().build(root, 16, 1)
            import pandas as pd
            df = pd.read_csv(csv)
            df = df[~((df.dataset=='d01') & (df.release=='old') & (df.algo=='RF'))]
            df.to_csv(csv, index=False)
            r = run('edge', '--inp', str(csv), '--edges', 'old:new', '--plan-dir', pl,
                    '--unblinded-exploratory', '--out', str(root/'edge.json'))
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('entirely absent', r.stderr)

if __name__ == '__main__':
    unittest.main()
