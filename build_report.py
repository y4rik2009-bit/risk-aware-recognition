"""Freeze completed records, create figures and an offline interactive explorer."""
import base64
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    raw=Path('results/raw'); verified=Path('results/verified'); docs=Path('docs')
    verified.mkdir(exist_ok=True,parents=True); docs.mkdir(exist_ok=True)
    for name in ['metrics.csv','summary.csv','run.json','history.json','calibration_audit.csv']:
        shutil.copyfile(raw/name,verified/name)
    d=pd.read_csv(raw/'metrics.csv')
    colors={'warm':'#8b97a7','bce':'#409aff','weighted':'#ba8cff','crt':'#fba35c','stopgrad':'#48c9b0'}
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for ax,condition in zip(axes,['clean','noise','occlusion']):
        for model in colors:
            sub=d[(d.condition==condition)&(d.model==model)&(d.control=='crc')]
            g=sub.groupby('alpha')[['fnr','fpr']].mean()
            ax.plot(g.fnr,g.fpr,'o-',label=model,color=colors[model])
        ax.set(title=condition,xlabel='Mean missed-label fraction',ylabel='Mean false-positive fraction')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(docs/'frontier.svg'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    sub=d[(d.alpha==.1)&(d.model=='crt')]
    for i,condition in enumerate(['clean','noise','occlusion']):
        for j,control in enumerate(['crc','crc_fresh']):
            v=sub[(sub.condition==condition)&(sub.control==control)]
            for ax,metric in zip(axes,['fnr','fpr']):
                ax.bar(i+j*.35,v[metric].mean(),width=.32,yerr=v[metric].std(),
                       color=['#fba35c','#48c9b0'][j],label=['Clean calibration','Fresh calibration'][j] if i==0 else None)
    for ax,metric in zip(axes,['Missed-label fraction','False-positive fraction']):
        ax.set_xticks(np.arange(3)+.175,['Clean','Noise','Occlusion']); ax.set_ylabel(metric); ax.grid(axis='y',alpha=.2)
    axes[0].axhline(.1,color='#9f3445',ls='--',label='Target 10%'); axes[0].legend(fontsize=8)
    fig.savefig(docs/'shift.svg'); plt.close(fig)
    audit=pd.read_csv(raw/'calibration_audit.csv')
    fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained'); ax.hist(audit.fnr,bins=20,color='#48b3a3',edgecolor='white')
    ax.axvline(.1,color='#a63446',ls='--',label='Target'); ax.axvline(audit.fnr.mean(),color='#274c63',label='Average over repeated calibrations')
    ax.set(xlabel='Test missed-label fraction per calibration experiment',ylabel='Count'); ax.legend(fontsize=8)
    fig.savefig(docs/'audit.svg'); plt.close(fig)
    demo=json.loads((raw/'demo.json').read_text()); n=24
    payload=dict(seed=demo['seed'],y=demo['y'][:n],images={},models={})
    for condition,images in demo['images'].items():
        payload['images'][condition]=base64.b64encode(np.array(images[:n],dtype='uint8').tobytes()).decode()
    for name,m in demo['models'].items():
        payload['models'][name]={k:np.round(v[:n],6).tolist() for k,v in m['test'].items()}
    # Stats use all 1500 test boards, examples use only the first 24 (seed 11).
    keys=['model','condition','alpha','control']
    aggregate=d.groupby(keys)[['fnr','fpr','cvar80','size','exact']].mean().reset_index()
    thresholds=d[d.seed==demo['seed']][keys+['threshold']]
    payload['records']=aggregate.merge(thresholds,on=keys).round(9).to_dict('records')
    payload['seed_count']=int(d.seed.nunique())
    (docs/'data.js').write_text('const EXPERIMENT = '+json.dumps(payload,separators=(',',':'))+';',encoding='utf-8')
    selected=d[(d.condition=='clean')&(d.control=='crc')&(d.alpha==.1)]
    print(selected.groupby('model')[['fnr','fpr','size']].mean().round(4).to_string())
    print('Audit mean FNR:',audit.fnr.mean(),'report bytes:',(docs/'data.js').stat().st_size)


if __name__=='__main__': main()
