"""VersionShift v2.2.2 regression tests for the newly discovered completeness/selection defects."""
import unittest, tempfile, json, hashlib, subprocess, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import analysis_v221 as A

class ExtraIntegrityTests(unittest.TestCase):
    def make_plan(self, d):
        p=Path(d); p.mkdir(exist_ok=True); plan={'datasets':[{'name':'a'},{'name':'b'}]}
        (p/'PLAN.json').write_text(json.dumps(plan))
        return p
    def rows(self,datasets=('a','b'),algs=A.ALGOS,releases=('old','new')):
        return pd.DataFrame([dict(dataset=d,release=r,policy='D',algo=a,rep=0,fold=f,status='ok',ba=.7)
            for d in datasets for r in releases for a in algs for f in range(5)])
    def test_missing_entire_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            plan=self.make_plan(Path(t)/'plan')
            with self.assertRaises(A.InvalidInput):
                A.validate_plan_coverage(self.rows(datasets=('a',)),plan,None,['old:new'],'D')
    def test_missing_entire_algorithm_job_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            plan=self.make_plan(Path(t)/'plan')
            with self.assertRaises(A.InvalidInput):
                A.validate_plan_coverage(self.rows(algs=A.ALGOS[:-1]),plan,None,['old:new'],'D')
    def test_complete_job_universe_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            plan=self.make_plan(Path(t)/'plan')
            result=A.validate_plan_coverage(self.rows(),plan,None,['old:new'],'D')
            self.assertEqual(result['required_jobs'],24)
    def test_selection_plan_hash_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            plan=self.make_plan(Path(t)/'plan');sel=Path(t)/'selection.json'
            sel.write_text(json.dumps({'stage_b_datasets':['a'],'plan_sha256':'bad'}))
            with self.assertRaises(A.InvalidInput):
                A.validate_plan_coverage(self.rows(datasets=('a',)),plan,sel,['old:new'],'D')
    def test_unblinded_no_plan_cannot_look_certified(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'rows.csv';self.rows(datasets=('a',)).to_csv(p,index=False)
            cmd=[sys.executable,str(ROOT/'code'/'analysis_v221.py'),'edge','--inp',str(p),'--edges','old:new','--unblinded-exploratory','--out',str(Path(t)/'o.json'),'--reps','1']
            r=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            rep=json.loads((Path(t)/'o.json').read_text())
            self.assertEqual(rep['plan_coverage']['status'],'NOT_CHECKED_EXPLORATORY')
    def test_no_plan_frozen_analysis_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'rows.csv';self.rows(datasets=('a',)).to_csv(p,index=False)
            r=subprocess.run([sys.executable,str(ROOT/'code'/'analysis_v221.py'),'edge','--inp',str(p),'--edges','old:new','--out',str(Path(t)/'o.json')],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)

if __name__=='__main__': unittest.main()
