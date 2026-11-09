#!/usr/bin/env python3
"""Read-only Stage B evidence summary for VersionShift v2.2.5.
Not a replacement for the frozen analysis or its G1-G7 gates.
"""
import argparse, hashlib, json, itertools
from pathlib import Path
import pandas as pd
import numpy as np

NAMES = ["EDGE_0213_0222", "EDGE_0213_0222_REVERSE",
         "EDGE_180_190", "EDGE_180_190_REVERSE"]
EDGES = [("EDGE_0213_0222", "0.21.3", "0.22.2.post1"),
         ("EDGE_180_190", "1.8.0", "1.9.1")]

def digest(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(2**20),b""): h.update(b)
    return h.hexdigest()

def fail(msg): raise SystemExit("STOP: "+str(msg))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs",required=True)
    ap.add_argument("--plan",required=True)
    ap.add_argument("--selection",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    runs,plan,sel,out=map(Path,(a.runs,a.plan,a.selection,a.out))
    out.mkdir(parents=True,exist_ok=True)
    pdict=json.loads((plan/"PLAN.json").read_text())
    sdict=json.loads(sel.read_text())
    chosen=set(sdict["stage_b_datasets"])
    if len(chosen)!=16: fail(f"expected 16 selected datasets, found {len(chosen)}")
    if sdict["plan_sha256"]!=digest(plan/"PLAN.json"): fail("selection/plan hash mismatch")
    expected={(d["name"],) for d in pdict["datasets"]}
    if not chosen <= {x[0] for x in expected}: fail("selected dataset not in plan")
    report={"plan_sha256":digest(plan/"PLAN.json"),
            "selection_sha256":digest(sel),"selected_datasets":sorted(chosen),
            "runs":{},"edge_d_descriptive":[],"warnings":[]}
    tables={}
    for name in NAMES:
        root=runs/name
        lk=json.loads((root/"RUN_LOCK.json").read_text())
        if lk["plan_sha256"] != report["plan_sha256"]: fail(f"{name}: plan lock mismatch")
        if set(lk.get("selected_datasets",[]))!=chosen: fail(f"{name}: selected dataset mismatch")
        if lk.get("selection_sha256")!=report["selection_sha256"]: fail(f"{name}: selection lock mismatch")
        summ=json.loads((root/"COLLECT_SUMMARY.json").read_text())
        if summ.get("missing_jobs"): fail(f"{name}: {len(summ['missing_jobs'])} missing jobs")
        if sum(summ.get("status_counts",{}).values())!=summ["planned_jobs"]:
            fail(f"{name}: status count != planned jobs")
        fpath=root/"CROSS_RELEASE_FOLDS.csv.gz"
        df=pd.read_csv(fpath)
        tables[name]=df
        good=df[df.status.eq("ok")].copy()
        if good.empty: fail(f"{name}: no successful fits")
        counts=good.groupby(["dataset","release","algo","policy"]).agg(
            rows=("ba","size"), fold_ids=("fold",lambda s: len(set(s)))).reset_index()
        invalid=counts[counts.rows.ne(30)]
        if len(invalid): fail(f"{name}: {len(invalid)} success jobs with !=30 folds")
        dup=good.duplicated(["dataset","release","algo","policy","rep","fold"]).sum()
        if dup: fail(f"{name}: {dup} duplicated fold identities")
        report["runs"][name]={"planned_jobs":summ["planned_jobs"],
            "status_counts":summ["status_counts"],
            "portability_counts":summ.get("portability_counts",{}),
            "successful_fold_rows":len(good),
            "run_lock_sha256":digest(root/"RUN_LOCK.json"),
            "collected_sha256":digest(fpath),
            "all_success_jobs_have_30_unique_folds":True}
    for name,old,new in EDGES:
        df=tables[name]
        base=df[(df.policy=="D") & df.status.eq("ok")].copy()
        expected_d=16*2*6
        actual_d=base[["dataset","release","algo"]].drop_duplicates().shape[0]
        if actual_d!=expected_d:
            report["warnings"].append(f"{name}: D success {actual_d}/{expected_d}; paired complete-case only")
        m=base.groupby(["dataset","release","algo"])["ba"].mean().unstack("release")
        if old not in m.columns or new not in m.columns: fail(f"{name}: missing version in D success")
        m["score_shift_pp"]=(m[new]-m[old])*100
        shifted=m.reset_index()
        shifted.to_csv(out/(name+"_D_algorithm_shifts.csv"),index=False)
        pivot=shifted.pivot(index="dataset", columns="algo",values="score_shift_pp")
        algos=sorted(set(pivot.columns))
        ds_rows=[]
        for ds,row in pivot.iterrows():
            if row.isna().any():
                continue
            shifts=[abs(row[x]-row[y]) for x,y in itertools.combinations(algos,2)]
            ds_rows.append({"dataset":ds,"M_pp":float(np.mean(shifts)),
                            "max_pair_pp":float(max(shifts)),
                            "affected_M_gt_0_5":int(np.mean(shifts)>.5),
                            "affected_max_gt_1":int(max(shifts)>1)})
        pd.DataFrame(ds_rows).to_csv(out/(name+"_D_dataset_effects.csv"),index=False)
        report["edge_d_descriptive"].append({
            "run":name,"old":old,"new":new,
            "D_job_success":actual_d,"D_jobs_expected":expected_d,
            "complete_case_datasets":len(ds_rows),
            "mean_M_pp":float(np.mean([r["M_pp"] for r in ds_rows])) if ds_rows else None,
            "affected_M_gt_0_5_count":sum(r["affected_M_gt_0_5"] for r in ds_rows),
            "affected_max_gt_1_count":sum(r["affected_max_gt_1"] for r in ds_rows)})
    target=out/"STAGE_B_READONLY_SUMMARY.json"
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report["edge_d_descriptive"],indent=2))
    print("Saved:",target)
    print("This is a descriptive independent check, not frozen inferential analysis.")
if __name__=="__main__":main()
