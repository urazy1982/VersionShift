#!/usr/bin/env python3
"""Standard-library self-consistency verification of *published aggregate summaries*.
Does not replay private y_true/y_pred or prove end-to-end experimental provenance.
"""
import argparse, json, csv, math, pathlib, sys

def readcsv(p):
    with p.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))

def check(root):
    results=root/'results'
    obj=json.loads((results/'validated_summary.json').read_text())
    groups=readcsv(results/'execution_group_inventory_33.csv')
    assert len(groups)==33
    for col,expected in [('expected',29472),('supported',23994),('unsupported',5312),('removed_behavior',166),('prediction_records_replayed',23994)]:
        got=sum(int(g[col]) for g in groups)
        assert got==expected, (col,got,expected)
    assert all(int(g['errors_count'])==0 for g in groups)
    assert all(g['audit_status']=='PASS_RAW_ARRAY_AND_TASK_STATISTICS_REPLAY' for g in groups)
    p=readcsv(results/'primary_two_edges.csv')
    assert len(p)==2
    target={
      'E_LINEAR_DUAL':(0.9953203700715383,0.00816922608448454),
      'E_ADA_SAMME':(2.0528140887726045,0.00029319451043094766)
    }
    for row in p:
        m,hol=target[row['edge_id']]
        assert int(row['n_tasks'])==48 and int(row['estimators'])==8
        assert row['frame']=='evaluation' and row['policy']=='D0' and row['preprocessing']=='P1'
        assert math.isclose(float(row['M_pp']),m,rel_tol=0,abs_tol=1e-12)
        assert math.isclose(float(row['p_holm']),hol,rel_tol=0,abs_tol=1e-14)
        assert float(row['lower_one_sided_pp'])>0.5
    hist=readcsv(results/'historical_63_task_level.csv')
    assert len(hist)==63 and len(set(r['task_id'] for r in hist))==63
    mean=sum(float(r['M_pp']) for r in hist)/len(hist)
    assert math.isclose(mean,5.028388095045953,abs_tol=1e-10)
    assert sum(r['winner_changed']=='True' for r in hist)==25
    arcs=readcsv(results/'historical_preprocessing_and_policy.csv')
    assert len(arcs)==9
    mapping={r['arm']:float(r['M_pp']) for r in arcs}
    assert math.isclose(mapping['E_ORIGINAL_P1_D0'],5.028388095045953,abs_tol=1e-10)
    assert math.isclose(mapping['E_ORIGINAL_P2_D0'],1.4642997681067895,abs_tol=1e-10)
    assert math.isclose(mapping['E_ORIGINAL_P1_D1'],0.4958305114787802,abs_tol=1e-10)
    assert math.isclose(mapping['E_ORIGINAL_P2_D1'],0.3399659723402404,abs_tol=1e-10)
    assert abs(mapping['E_ORIGINAL_P1_D0']-mapping['E_ORIGINAL_FIXED_STACK_P1_D0'])<1e-12
    assert abs(mapping['E_ORIGINAL_P2_D0']-mapping['E_ORIGINAL_FIXED_STACK_P2_D0'])<1e-12
    s=obj['job_accounting']
    assert s['expected']==s['observed']==29472 and s['groups']==33
    assert s['fold_scores_replayed']==719820
    assert s['statuses']=={'SUPPORTED':23994,'UNSUPPORTED':5312,'REMOVED_BEHAVIOR':166}
    assert len(readcsv(results/'previous_v225_stageB.csv'))==8
    print(json.dumps({'status':'PUBLIC_AGGREGATE_SELF_CONSISTENCY_PASS','groups':33,'jobs':29472,'primary_edges':2,'historical_tasks':63,'historical_M_pp':mean,'historical_winner_changes':25},indent=2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1])
    args=ap.parse_args()
    try:check(args.root.resolve())
    except (AssertionError,ValueError,KeyError,OSError) as exc:
        print('SELF_CONSISTENCY_FAIL: '+str(exc),file=sys.stderr);sys.exit(1)
if __name__=='__main__':main()
