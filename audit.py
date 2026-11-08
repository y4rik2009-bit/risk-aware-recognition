"""Repeat calibration on independent generated draws for one frozen model."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data import source_pools, boards
from model import Recognizer, probabilities
from risk import crc_threshold, metrics


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',default='results/raw'); p.add_argument('--repeats',type=int,default=200)
    a=p.parse_args(); root=Path(a.run); manifest=json.loads((root/'run.json').read_text())
    seed=manifest['seeds'][0]; torch.set_num_threads(4)
    model=Recognizer(); model.load_state_dict(torch.load(root/'crt.pt',weights_only=True))
    images,labels,_,pool=source_pools(seed); rows=[]
    for repeat in range(a.repeats):
        x,y=boards(images,labels,pool,500,100000+repeat)
        prob=probabilities(model,x)
        q=crc_threshold(prob[:200],y[:200],0.1)
        rows.append(dict(repeat=repeat,threshold=q,**metrics(prob[200:],y[200:],q)))
    pd.DataFrame(rows).to_csv(root/'calibration_audit.csv',index=False)
    print(pd.DataFrame(rows)[['fnr','fpr']].agg(['mean','std']).to_string())


if __name__=='__main__': main()
