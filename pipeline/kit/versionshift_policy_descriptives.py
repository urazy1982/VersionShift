#!/usr/bin/env python3
"""Read-only secondary policy comparison for collected VersionShift Stage B jobs."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--project', required=True)
ap.add_argument('--positive', required=True)
ap.add_argument('--later', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
sys.path.insert(0, str(Path(a.project)/'code'))
import analysis_v221 as AV

report={}
for f, old, new in [(a.positive,'0.21.3','0.22.2.post1'),(a.later,'1.8.0','1.9.1')]:
    df=pd.read_csv(f)
    ok=AV.validate(df,6,5)
    row={}
    for policy in ('D','E','P','S'):
        mat,_,_=AV.shift_matrix(ok,old,new,policy,policy)
        x=mat.to_numpy(dtype=float)
        complete=np.isfinite(x).all(axis=1)
        if not complete.any():
            row[policy]={'complete_case_datasets':0,'mean_M_pp':None,'affected_M_gt_0_5':None}
            continue
        M=np.abs(x[complete]).mean(axis=1)
        row[policy]={'complete_case_datasets':int(complete.sum()),'mean_M_pp':float(M.mean()),
                     'affected_M_gt_0_5':int((M>0.5).sum())}
    report[old+':'+new]=row
Path(a.out).write_text(json.dumps(report,indent=2))
print('SECONDARY POLICY DESCRIPTIVES:',json.dumps(report))
