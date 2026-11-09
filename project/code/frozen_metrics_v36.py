"""Fixed evaluator: NumPy/SciPy only; never imports scikit-learn."""
import numpy as np
from scipy import stats


def binary_auc(y, score):
    y,score=np.asarray(y),np.asarray(score,dtype=float)
    if y.ndim!=1 or score.shape!=y.shape or not np.isfinite(score).all():
        raise ValueError('invalid binary AUC inputs')
    n1=int((y==1).sum()); n0=int((y==0).sum())
    if n1+n0!=len(y) or not n1 or not n0:
        raise ValueError('AUC requires both classes and labels 0/1')
    ranks=stats.rankdata(score,method='average')
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def metrics(arrays, score_kind):
    y,pred,score,classes=(arrays[k] for k in ('y_true','y_pred','scores','classes'))
    k=len(classes)
    if not np.array_equal(classes,np.arange(k)) or not np.isin(y,classes).all() or not np.isin(pred,classes).all():
        raise ValueError('class mapping mismatch')
    if k==2:
        auc=binary_auc(y, score if score.ndim==1 else score[:,1])
    else:
        if score.shape!=(len(y),k):raise ValueError('multiclass scores must have one column per class')
        auc=float(np.mean([binary_auc((y==c).astype(int),score[:,c]) for c in classes]))
    recalls=[];f1=[]
    for c in classes:
        tp=int(((y==c)&(pred==c)).sum());fn=int(((y==c)&(pred!=c)).sum());fp=int(((y!=c)&(pred==c)).sum())
        recalls.append(tp/(tp+fn) if tp+fn else 0.)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    out=dict(auc=auc,balanced_accuracy=float(np.mean(recalls)),macro_f1=float(np.mean(f1)),recalls=recalls)
    if score_kind=='predict_proba':
        if score.shape!=(len(y),k) or (score<0).any() or (score>1).any() or not np.allclose(score.sum(1),1):
            raise ValueError('invalid probabilities')
        order=np.argsort(score.max(1),kind='stable')
        out['ece']=float(sum(len(ix)/len(y)*abs(float((pred[ix]==y[ix]).mean())-float(score[ix].max(1).mean()))
                             for ix in np.array_split(order,min(10,len(y))) if len(ix)))
    return out


def posterior(differences,rope=.01,threshold=.95):
    x=np.asarray(differences,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all() or not np.isfinite(rope) or rope<0 or not 0.5<threshold<1:
        raise ValueError('invalid posterior input')
    mean=float(x.mean());variance=float(x.var(ddof=1))
    if np.all(x==x[0]):
        left,eq,right=(1.,0.,0.) if mean < -rope else ((0.,0.,1.) if mean>rope else (0.,1.,0.))
        direction=1. if mean>0 else (0. if mean<0 else .5)
    else:
        scale=float(np.sqrt((1/len(x)+.2/.8)*variance))
        dist=stats.t(df=len(x)-1,loc=mean,scale=scale)
        left=float(dist.cdf(-rope));right=float(dist.sf(rope));eq=max(0.,1-left-right)
        direction=float(dist.sf(0))
    state='A>B' if right>=threshold else ('B>A' if left>=threshold else ('A~B' if eq>=threshold else 'U'))
    return dict(state=state,p_left=left,p_equivalent=eq,p_right=right,p_direction=direction,mean=mean)


def s_c2(p1,p2):
    """C2 posterior-movement score (Study Design v1.2.1/v1.3): change in the posterior mass
    beyond +delta plus change in the mass beyond -delta between versions."""
    return abs(p2['p_right']-p1['p_right'])+abs(p2['p_left']-p1['p_left'])


def reversal(a,b):
    return (a=='A>B' and b=='B>A') or (a=='B>A' and b=='A>B')


def cluster_interval(rows,field,denominator,replicates=2000,seed=20261103):
    """Ratio of sums; resample whole datasets, including all arms/pairs/edges."""
    ids=sorted({r['task'] for r in rows})
    by=[(sum(r[field] for r in rows if r['task']==d),sum(r[denominator] for r in rows if r['task']==d)) for d in ids]
    total=sum(b for a,b in by)
    if not total:return dict(estimate=None,interval=None,reason='no eligible comparisons')
    estimate=sum(a for a,b in by)/total
    if len(ids)<2 or all(a==0 for a,b in by):
        return dict(estimate=estimate,interval=None,reason='sparse/zero events; no degenerate bootstrap interval reported')
    rng=np.random.default_rng(seed);values=[]
    for _ in range(replicates):
        selected=rng.integers(0,len(by),len(by));num=sum(by[i][0] for i in selected);den=sum(by[i][1] for i in selected)
        if den:values.append(num/den)
    return dict(estimate=estimate,interval=[float(v) for v in np.quantile(values,[.025,.975])],clusters=len(ids))


def korn_graubard(events, n, level=.95):
    """Korn-Graubard interval for a clustered proportion (v0.2.0).

    events, n: per-dataset counts. Ratio-of-sums estimate; cluster-robust variance across
    datasets gives an effective sample size, which is t-adjusted (K-1 df) and used in a
    Clopper-Pearson interval. Returns None with zero events or fewer than two clusters:
    the zero-event case is reported with the separate dataset-any bound instead.
    """
    from scipy.stats import beta, norm, t
    e=np.asarray(events,dtype=float);m=np.asarray(n,dtype=float);k=len(m);N=m.sum();R=e.sum()
    if k<2 or N<=0 or R<=0:return None
    p=R/N
    if p>=1:return None
    v=k/(k-1)*float(((e-p*m)**2).sum())/N**2
    deff=max(v/(p*(1-p)/N),1.0)
    alpha=1-level
    ne=N/deff*(norm.ppf(1-alpha/2)/t.ppf(1-alpha/2,k-1))**2
    x=p*ne
    lo=float(beta.ppf(alpha/2,x,ne-x+1));hi=float(beta.ppf(1-alpha/2,x+1,ne-x))
    return (lo,hi)


def cluster_kg(rows,field,denominator):
    ids=sorted({r['task'] for r in rows})
    ev=[sum(r[field] for r in rows if r['task']==d) for d in ids]
    nn=[sum(r[denominator] for r in rows if r['task']==d) for d in ids]
    if not sum(nn):return dict(estimate=None,interval=None,reason='no eligible comparisons')
    ci=korn_graubard(ev,nn)
    return dict(estimate=sum(ev)/sum(nn),interval=list(ci) if ci else None,clusters=len(ids),
                method='Korn-Graubard (cluster-robust effective n, t-adjusted Clopper-Pearson)',
                reason=None if ci else 'zero events or fewer than two clusters; see dataset-any bound')
