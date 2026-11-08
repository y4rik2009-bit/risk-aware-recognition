import copy
import time
import numpy as np
import torch
from torch.nn import functional as F
from benchmark.networks import LabelNet,Linear,asymmetric_loss,label_neighbors,predict
from benchmark.evaluation import validation_utility
from risk import differentiable_threshold


def fit(model,x,y,xv,yv,epochs,seed,mode='bce',lr=.001):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    x,y=torch.as_tensor(x),torch.as_tensor(y)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.001)
    best=copy.deepcopy(model.state_dict()); best_score=validation_utility(predict(model,xv),yv)
    history=[]; t0=time.perf_counter()
    for epoch in range(epochs):
        model.train(); order=rng.permutation(len(x)); total=0.; batches=0
        for start in range(0,len(x),512):
            idx=order[start:start+512]
            if len(idx)<32: continue
            logits=model(x[idx]); target=y[idx]
            if mode in ['crt','stopgrad']:
                p=logits.sigmoid(); half=len(p)//2
                q=differentiable_threshold(p[:half],target[:half],.1)
                if mode=='stopgrad': q=q.detach()
                soft=torch.sigmoid((p[half:]-q)/.05)
                fpr=(soft*(1-target[half:])).sum(1)/(1-target[half:]).sum(1).clamp_min(1)
                cost=fpr.mean()+.3*F.binary_cross_entropy_with_logits(logits,target)
            elif mode=='asl': cost=asymmetric_loss(logits,target)
            else: cost=F.binary_cross_entropy_with_logits(logits,target)
            optimizer.zero_grad(); cost.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step(); total+=float(cost.detach()); batches+=1
        if (epoch+1)%5==0 or epoch+1==epochs:
            p=predict(model,xv); score=validation_utility(p,yv)
            history.append(dict(epoch=epoch+1,objective=total/max(batches,1),validation_utility=score))
            if score<best_score:
                best_score=score; best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model,history,best_score,time.perf_counter()-t0


def train_all(x,y,xv,yv,seed,warm_epochs=40,fine_epochs=40):
    neighbors=label_neighbors(torch.tensor(y)); trials=[]; winner=None
    # Fixed search space. Every candidate sees training and validation only.
    for width,lr,dropout in [(128,.001,.1),(256,.001,.15),(256,.0003,.1)]:
        torch.manual_seed(seed)
        model=LabelNet(x.shape[1],y.shape[1],width,dropout,neighbors)
        model,h,score,seconds=fit(model,x,y,xv,yv,warm_epochs,seed,lr=lr)
        trials.append(dict(width=width,lr=lr,dropout=dropout,score=score,seconds=seconds,history=h))
        if winner is None or score<winner[2]: winner=(model,(width,lr,dropout),score)
    base,config,_=winner
    results={}; total_search=sum(t['seconds'] for t in trials)
    for name,mode in [('residual_bce','bce'),('asymmetric','asl'),('risk_training','crt'),('no_threshold_grad','stopgrad'),('label_graph','bce')]:
        model=copy.deepcopy(base); model.use_graph=name=='label_graph'
        model,h,score,seconds=fit(model,x,y,xv,yv,fine_epochs,seed+100,mode=mode,lr=config[1])
        results[name]=(model,dict(history=h,validation_score=score,fit_seconds=seconds,
                                  shared_search_seconds=total_search,parameters=sum(p.numel() for p in model.parameters()),
                                  width=config[0],dropout=config[2],lr=config[1]))
    torch.manual_seed(seed)
    linear,h,score,seconds=fit(Linear(x.shape[1],y.shape[1]),x,y,xv,yv,warm_epochs+fine_epochs,seed)
    results['linear']=(linear,dict(history=h,validation_score=score,fit_seconds=seconds,
                                  shared_search_seconds=0,parameters=sum(p.numel() for p in linear.parameters())))
    return results,trials
