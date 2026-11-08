import numpy as np
from sklearn.metrics import average_precision_score
from risk import crc_threshold,corc_threshold,choose_t,empirical_cvar
from benchmark.calibration import RiskCurve,select_t


def label_thresholds(p,y,alpha):
    thresholds=[]
    for j in range(y.shape[1]):
        positive=y[:,j]>0
        thresholds.append(crc_threshold(p[positive,j,None],np.ones((positive.sum(),1)),alpha) if positive.any() else 0.)
    return np.array(thresholds)


def risk_metrics(p,y,q):
    selected=p>=q; positive=y.sum(1); negative=(1-y).sum(1)
    loss=(y*(~selected)).sum(1)/positive
    fp=((1-y)*selected).sum(1)
    fn_by_label=(y*(~selected)).sum(0); support=y.sum(0)
    valid=support>0; label_fnr=fn_by_label[valid]/support[valid]
    tp=(y*selected).sum()
    return dict(fnr=float(loss.mean()),fpr=float((fp/np.maximum(negative,1)).mean()),
                cvar80=empirical_cvar(loss),set_size=float(selected.sum(1).mean()),
                precision=float(tp/max(selected.sum(),1)),micro_recall=float(tp/y.sum()),
                macro_label_fnr=float(label_fnr.mean()),worst_label_fnr=float(label_fnr.max()),
                extra_labels=float(fp.mean()),evaluated_labels=int(valid.sum()))


def ranking_metrics(p,y):
    valid=y.sum(0)>0
    return dict(macro_ap=float(average_precision_score(y[:,valid],p[:,valid],average='macro')),
                micro_ap=float(average_precision_score(y,p,average='micro')),
                ap_labels=int(valid.sum()))


def validation_utility(p,y):
    # Both folds are inside validation; final calibration and test are untouched.
    n=len(p)//2; scores=[]
    for c,e in [(slice(None,n),slice(n,None)),(slice(n,None),slice(None,n))]:
        q=crc_threshold(p[c],y[c],.1); m=risk_metrics(p[e],y[e],q)
        scores.append(m['fpr']+2*max(m['fnr']-.1,0))
    return float(np.mean(scores))


def calibrators(p,y,pv,yv,alpha):
    curve=RiskCurve(p,y); validation_curve=RiskCurve(pv,yv)
    t=select_t(validation_curve,alpha,.8)
    return [('fixed',.5,None),('mean_crc',curve.threshold(alpha),None),
            ('tail_cvar',curve.threshold(alpha,.8,t),t),
            ('label_crc',label_thresholds(p,y,alpha),None)]


def group_diagnostics(p,y,q,training_counts,names):
    pred=p>=q; order=np.argsort(training_counts); rare=set(order[:max(1,len(order)//3)])
    rows=[]
    for j,name in enumerate(names):
        n=int(y[:,j].sum()); fn=int((y[:,j]*(~pred[:,j])).sum())
        rows.append(dict(label=name,train_positives=int(training_counts[j]),test_positives=n,
                         fnr=fn/n if n else None,rare=j in rare,predicted=int(pred[:,j].sum())))
    return rows
