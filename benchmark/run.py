"""Real-data experiments. Every completed dataset/seed is an atomic resumable unit."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn import config_context
from benchmark.datasets import DATASETS,load
from benchmark.protocol import deduplicate,partition,Preprocessor,shifted
from benchmark.training import train_all
from benchmark.networks import predict
from benchmark.evaluation import calibrators,risk_metrics,ranking_metrics,validation_utility,group_diagnostics


def fingerprint(config):
    h=hashlib.sha256(json.dumps(config,sort_keys=True).encode())
    for p in sorted(Path('benchmark').glob('*.py')): h.update(p.read_bytes())
    h.update(Path('risk.py').read_bytes())
    return h.hexdigest()


def neighbor_predictions(x,y,queries,seed):
    t0=time.perf_counter(); knn=NearestNeighbors(n_neighbors=min(31,len(x)),metric='cosine',n_jobs=4).fit(x)
    output={}
    with config_context(working_memory=128):
        for split,q in queries.items():
            distances,indices=knn.kneighbors(q)
            output[split]={}
            for k in [5,15,31]:
                w=1/np.maximum(distances[:,:k],1e-5)
                output[split][k]=(y[indices[:,:k]]*w[:,:,None]).sum(1)/w.sum(1,keepdims=True)
    k=min([5,15,31],key=lambda k:validation_utility(output['validation'][k],queries['yv'])) if False else None
    return output,time.perf_counter()-t0


def run_unit(name,seed,a,config,code_hash):
    folder=Path(a.out)/'raw'/f'{name}-{seed}'; folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'complete.json'
    if marker.exists():
        if json.loads(marker.read_text())['fingerprint']!=code_hash: raise ValueError('Resume fingerprint changed; use a new output folder')
        print('resume',name,seed,flush=True); return
    started=time.perf_counter(); cpu=time.process_time()
    x,y,meta=load(name); x,y,duplicates=deduplicate(x,y)
    parts=partition(len(x),seed); prep=Preprocessor().fit(x[parts['train']])
    data={key:(prep.transform(x[idx]),y[idx]) for key,idx in parts.items()}
    xt,yt=data['train']; xv,yv=data['validation']; xc,yc=data['calibration']; xe,ye=data['test']
    result,trials=train_all(xt,yt,xv,yv,seed,a.warm_epochs,a.fine_epochs)
    rows=[]; diagnostics=[]; model_info={}; cases={}; saved={}
    for model,(network,info) in result.items():
        predictions={key:predict(network,xx) for key,(xx,_) in data.items() if key!='train'}
        condition_preds={'clean':(predictions['calibration'],predictions['validation'],predictions['test'])}
        for condition in ['dropout','noise']:
            condition_preds[condition]=(predict(network,shifted(xc,condition,seed+1000)),
                                         predict(network,shifted(xv,condition,seed+2000)),
                                         predict(network,shifted(xe,condition,seed+3000)))
        model_info[model]=info
        saved[model]=condition_preds
        torch.save(dict(state=network.state_dict(),model=model,features=x.shape[1],labels=y.shape[1],info=info),folder/f'{model}.pt')
    # k-NN is a separate non-neural baseline, selected on validation only.
    tknn=time.perf_counter(); neighbor=NearestNeighbors(n_neighbors=min(31,len(xt)),metric='cosine',n_jobs=4).fit(xt)
    def knn_scores(xx,k):
        with config_context(working_memory=128): distances,indices=neighbor.kneighbors(xx)
        w=1/np.maximum(distances[:,:k],1e-5)
        return np.clip((yt[indices[:,:k]]*w[:,:,None]).sum(1)/w.sum(1,keepdims=True),0,1).astype('float32')
    selected_k=min([5,15,31],key=lambda k:validation_utility(knn_scores(xv,k),yv))
    saved['knn']={condition:tuple(knn_scores(shifted(xx,condition,seed+offset),selected_k)
                                  for xx,offset in [(xc,1000),(xv,2000),(xe,3000)])
                  for condition in ['clean','dropout','noise']}
    model_info['knn']=dict(k=selected_k,fit_and_prediction_seconds=time.perf_counter()-tknn,parameters=0)
    for model,conditions in saved.items():
        pc,pv,_=conditions['clean']
        for condition,(fresh_cal,fresh_val,pt) in conditions.items():
            rank=ranking_metrics(pt,ye)
            for alpha in [.05,.1,.2]:
                controls=calibrators(pc,yc,pv,yv,alpha)
                fresh=calibrators(fresh_cal,yc,fresh_val,yv,alpha)
                if condition!='clean': controls += [(n+'_fresh',q,t) for n,q,t in fresh if n in ['mean_crc','tail_cvar']]
                for control,q,t in controls:
                    rows.append(dict(dataset=name,seed=seed,model=model,condition=condition,alpha=alpha,
                                     control=control,t=t,**risk_metrics(pt,ye,q),**rank))
                    if condition=='clean' and alpha==.1 and control in ['mean_crc','label_crc']:
                        diagnostics += [dict(dataset=name,seed=seed,model=model,control=control,**r)
                                        for r in group_diagnostics(pt,ye,q,yt.sum(0),meta['labels'])]
                    if alpha==.1 and control in ['mean_crc','mean_crc_fresh','label_crc']:
                        cases[f'{model}/{condition}/{control}']=dict(q=np.asarray(q).tolist(),
                            probabilities=pt[:12].round(5).tolist(),truth=ye[:12].astype(int).tolist())
        if model!='knn': np.savez_compressed(folder/f'{model}_calibration.npz',p=pc,y=yc,pv=pv,yv=yv)
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(folder/'labels.csv',index=False)
    np.savez_compressed(folder/'preprocessor.npz',median=prep.median,mean=prep.mean,scale=prep.scale)
    (folder/'cases.json').write_text(json.dumps(cases,separators=(',',':')))
    (folder/'splits.json').write_text(json.dumps({k:v.tolist() for k,v in parts.items()}))
    manifest=dict(fingerprint=code_hash,dataset=meta,seed=seed,duplicates_removed=duplicates,rows=len(x),
                  partition_sizes={k:len(v) for k,v in parts.items()},model_info=model_info,search=trials,
                  wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
                  split_sha256=hashlib.sha256((folder/'splits.json').read_bytes()).hexdigest())
    marker.write_text(json.dumps(manifest,indent=2)); print(name,seed,'complete',round(manifest['wall_seconds'],1),'seconds',flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--datasets',nargs='+',default=DATASETS)
    p.add_argument('--seeds',nargs='+',type=int,default=[11,23,42,71,101]); p.add_argument('--warm-epochs',type=int,default=40)
    p.add_argument('--fine-epochs',type=int,default=40); p.add_argument('--threads',type=int,default=8)
    p.add_argument('--out',default='results/research'); a=p.parse_args(); torch.set_num_threads(a.threads)
    config=vars(a); code_hash=fingerprint(config); root=Path(a.out); root.mkdir(parents=True,exist_ok=True)
    (root/'config.json').write_text(json.dumps(dict(config=config,fingerprint=code_hash,python=platform.python_version(),
        torch=torch.__version__,numpy=np.__version__,logical_cpus=os.cpu_count()),indent=2))
    for name in a.datasets:
        for seed in a.seeds: run_unit(name,seed,a,config,code_hash)


if __name__=='__main__': main()
