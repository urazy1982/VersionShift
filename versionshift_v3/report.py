"""Read-only summaries retaining denominators, portability and inference status."""
import argparse,glob,json
from pathlib import Path
from .utils import read_json

def render_reports(reports):
    lines=['# VersionShift audited CPU results','',
           'Finite-frame evidence on the historically exposed allocated tasks. t, bootstrap and iid-binomial quantities are task-model sensitivity analyses, not finite-census uncertainty or unconditional population equivalence. Secondary summaries are descriptive. Missing/nonportable cells are not zero drift.','',
           '| Run | Policy and panel | Complete / selected | M pp | Task-model sensitivity lower pp | Conditional test status | Conditional equivalence |','|---|---|---:|---:|---:|---|---|']
    for name,r in reports:
        for policy,panels in r.get('by_policy',{}).items():
            for panel,v in panels.items():
                lines.append('| {} | {} / {} | {} / {} | {} | {} | {} | {} |'.format(name,policy,panel,v.get('K_complete',0),v.get('selected_denominator'),v.get('M_pp'),v.get('one_sided_lower95_pp'),v.get('primary_test',{}).get('status',v.get('status')),v.get('equivalence',{}).get('status','NOT_ESTIMABLE')))
        lines+=['',f'Run {name}: portability and failure denominators','', '```json',json.dumps(r.get('portability_by_policy',{}),indent=2),'```','', 'Paired mitigation denominators','', '```json',json.dumps(r.get('mitigation',{}),indent=2),'```','']
        lines+=['Full signed-pair intervals, prevalence bounds and assumptions, margin-qualified winner denominators, material reversals, estimator shifts, leverage, prediction mismatches and warning/convergence diagnostics are retained in the linked machine audit: '+name+'.','']
    return '\n'.join(lines)+'\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('--audits',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    paths=sorted({p for pattern in a.audits for p in glob.glob(pattern)});out=Path(a.out)
    if out.exists():raise SystemExit('STOP report already exists')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(render_reports([(p,read_json(p)) for p in paths]));print('REPORT_WRITTEN',len(paths))
if __name__=='__main__':main()
