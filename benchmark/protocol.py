import hashlib
import numpy as np


def deduplicate(x,y):
    seen=set(); indices=[]
    for i,(a,b) in enumerate(zip(x,y)):
        key=hashlib.sha256(a.tobytes()+b.tobytes()).digest()
        if key not in seen: seen.add(key); indices.append(i)
    return x[indices],y[indices],len(x)-len(indices)


def partition(n,seed):
    indices=np.random.default_rng(seed).permutation(n)
    a,b,c=int(n*.55),int(n*.70),int(n*.85)
    return dict(train=indices[:a],validation=indices[a:b],calibration=indices[b:c],test=indices[c:])


class Preprocessor:
    def fit(self,x):
        self.median=np.nanmedian(x,axis=0); self.median=np.nan_to_num(self.median)
        filled=np.where(np.isfinite(x),x,self.median)
        self.mean=filled.mean(0); self.scale=filled.std(0); self.scale[self.scale<1e-6]=1
        return self
    def transform(self,x):
        return np.clip((np.where(np.isfinite(x),x,self.median)-self.mean)/self.scale,-10,10).astype('float32')


def shifted(x,condition,seed):
    rng=np.random.default_rng(seed)
    if condition=='clean': return x
    if condition=='dropout': return (x*(rng.random(x.shape)>.3)).astype('float32')
    if condition=='noise': return (x+rng.normal(0,.5,x.shape)).astype('float32')
    raise ValueError(condition)
