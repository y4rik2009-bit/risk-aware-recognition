"""Run a trained model on one 20x20 board with its paired calibration data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from model import Recognizer, probabilities
from risk import crc_threshold, choose_t, corc_threshold


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='results/raw')
    parser.add_argument('--model',default='crt',choices=['crt','warm','bce','weighted','stopgrad'])
    parser.add_argument('--image',required=True,help='A .npy array of shape (20,20), values in [0,1]')
    parser.add_argument('--alpha',type=float,default=.1)
    parser.add_argument('--tail',action='store_true')
    a=parser.parse_args(); root=Path(a.run)
    x=np.load(a.image,allow_pickle=False)
    if x.shape!=(20,20) or not np.isfinite(x).all() or np.any((x<0)|(x>1)):
        raise ValueError('Image must be a finite 20x20 array with values in [0,1]')
    metadata=json.loads((root/'checkpoint_hashes.json').read_text())
    checkpoint=root/f'{a.model}.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=metadata[a.model]:
        raise ValueError('Checkpoint does not match the saved calibration scores')
    demo=json.loads((root/'demo.json').read_text())
    cal=demo['models'][a.model]; p,y=np.array(cal['cal']),np.array(demo['cal_y'])
    if a.tail:
        t=choose_t(np.array(cal['validation']),np.array(demo['val_y']),a.alpha,.8)
        threshold=corc_threshold(p,y,a.alpha,.8,t)
    else:
        threshold=crc_threshold(p,y,a.alpha)
    model=Recognizer(); model.load_state_dict(torch.load(checkpoint,weights_only=True))
    scores=probabilities(model,x.reshape(1,-1).astype('float32'))[0]
    print(json.dumps(dict(labels=np.flatnonzero(scores>=threshold).tolist(),threshold=threshold,
                          probabilities=scores.tolist(),note='Risk control requires inputs from the calibration distribution.'),indent=2))


if __name__=='__main__': main()
