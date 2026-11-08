"""Exact risk curves in O(number of positive labels * log N), not O(N*L*iterations)."""
import numpy as np
from risk import validate


class RiskCurve:
    def __init__(self,p,y):
        p,y=validate(p,y); self.n=len(y)
        row,col=np.nonzero(y)
        scores=p[row,col]
        order=np.lexsort((scores,row)); row=row[order]; scores=scores[order]
        counts=y.sum(1); starts=np.r_[0,np.cumsum(counts.astype(int))[:-1]]
        rank=np.arange(len(row))-starts[row]
        self.before=rank/counts[row]; self.after=(rank+1)/counts[row]
        self.scores=scores
        self.global_order=np.argsort(scores,kind='stable')
        ordered=scores[self.global_order]
        self.ends=np.r_[np.flatnonzero(ordered[1:]!=ordered[:-1]),len(ordered)-1]
        self.values=ordered[self.ends]

    def threshold(self,alpha,delta=None,t=0.):
        if not 0<alpha<1: raise ValueError('alpha outside (0,1)')
        if delta is None:
            weights=self.after-self.before; budget=alpha*(self.n+1)-1
        else:
            if not 0<=delta<1 or not 0<=t<=alpha: raise ValueError('Invalid tail parameters')
            weights=(np.maximum(self.after-t,0)-np.maximum(self.before-t,0))/(1-delta)
            upper=t+(1-t)/(1-delta)
            budget=alpha*(self.n+1)-self.n*t-upper
        if budget<0: return 0.
        cumulative=np.cumsum(weights[self.global_order])[self.ends]
        blocked=np.flatnonzero(cumulative>budget+1e-10)
        return float(self.values[blocked[0]]) if len(blocked) else 1.


def select_t(curve,alpha,delta=.8):
    return float(max(np.linspace(0,alpha,9),key=lambda t:curve.threshold(alpha,delta,t)))
