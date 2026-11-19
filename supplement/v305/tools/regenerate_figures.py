#!/usr/bin/env python3
"""Optionally render summary figures with matplotlib; output is illustrative only."""
import csv, pathlib, argparse

def rows(path):
 with open(path,newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);args=p.parse_args()
 root=args.root.resolve();out=root/'figures';out.mkdir(exist_ok=True)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 primary=rows(root/'results/primary_two_edges.csv')
 fig,ax=plt.subplots(figsize=(8,4.2))
 y=[float(r['M_pp']) for r in primary]
 x=range(len(primary));ax.bar(x,y,yerr=[[v-float(r['bootstrap95_low_pp']) for v,r in zip(y,primary)],[float(r['bootstrap95_high_pp'])-v for v,r in zip(y,primary)]],capsize=5)
 ax.axhline(.5,ls='--',lw=1,color='gray',label='Preregistered 0.5 pp threshold')
 ax.set_xticks(list(x),['1.4.2 → 1.5.2\nLinearSVC','1.5.2 → 1.6.1\nAdaBoost']);ax.set_ylabel('Mean pairwise comparison drift M (pp)')
 ax.set_title('Modern release edges: finite-frame primary results');ax.legend();fig.tight_layout();fig.savefig(out/'primary_two_edges.png',dpi=170);plt.close(fig)
 hist=rows(root/'results/historical_preprocessing_and_policy.csv')
 lookup={r['arm']:float(r['M_pp']) for r in hist}
 labels=['Archived P0 / D0','P1 / D0','P2 / D0','P1 / D1','P2 / D1'];keys=[None,'E_ORIGINAL_P1_D0','E_ORIGINAL_P2_D0','E_ORIGINAL_P1_D1','E_ORIGINAL_P2_D1']
 values=[5.006712451138972 if k is None else lookup[k] for k in keys]
 fig,ax=plt.subplots(figsize=(8.5,4.4));ax.bar(labels,values);ax.set_ylabel('Historical mean M (pp)');ax.set_title('Historical upgrade: preprocessing and config sensitivity');ax.tick_params(axis='x',labelrotation=15);fig.tight_layout();fig.savefig(out/'historical_sensitivity.png',dpi=170);plt.close(fig)
 g=rows(root/'results/execution_group_inventory_33.csv');cnt=[sum(int(a[k]) for a in g) for k in ['supported','unsupported','removed_behavior']]
 fig,ax=plt.subplots(figsize=(7.7,4.2));bars=ax.bar(['Supported','Unsupported','Removed behavior'],cnt)
 ax.bar_label(bars,padding=3);ax.set_ylim(0,26000);ax.set_ylabel('Registered execution records');ax.set_title('Complete 29,472-record status ledger');fig.tight_layout();fig.savefig(out/'typed_job_inventory.png',dpi=170);plt.close(fig)
 print('FIGURES_RENDERED',len(list(out.glob('*.png'))))
if __name__=='__main__':main()
