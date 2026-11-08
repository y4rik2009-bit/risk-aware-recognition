import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import time
from pathlib import Path
import numpy as np

DATASETS = ['Scene','Yeast','Emotions','Bibtex','Mediamill','Delicious']
SOURCE = 'https://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/'


def parse_arff(text, label_names):
    attributes, lines, data = [], [], False
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith('%'): continue
        if data: lines.append(line)
        elif line.lower().startswith('@attribute'):
            match=re.match(r'@attribute\s+(?:"([^"]+)"|\x27([^\x27]+)\x27|(\S+))',line,re.I)
            attributes.append(next(v for v in match.groups() if v is not None))
        elif line.lower().startswith('@data'): data=True
    a=np.zeros((len(lines),len(attributes)),dtype=np.float32)
    for i,line in enumerate(lines):
        if line.startswith('{'):
            for entry in line.strip('{} ').split(','):
                if not entry.strip(): continue
                key,value=entry.strip().split(None,1)
                a[i,int(key)]=float(value.strip("'\"")) if value!='?' else np.nan
        else:
            values=next(csv.reader([line]))
            if len(values)!=len(attributes): raise ValueError('ARFF width mismatch')
            a[i]=[float(v.strip("'\"")) if v!='?' else np.nan for v in values]
    indices=[attributes.index(n) for n in label_names]
    feature_indices=[j for j in range(len(attributes)) if j not in indices]
    y=a[:,indices]
    if not np.isin(y,[0,1]).all(): raise ValueError('Labels are not binary')
    return a[:,feature_indices],y,[attributes[j] for j in feature_indices]


def load(name, cache='.cache/multilabel'):
    if name not in DATASETS: raise ValueError(name)
    root=Path(cache); root.mkdir(parents=True,exist_ok=True)
    archive=root/f'{name}.zip'; parsed=root/f'{name}.npz'
    if not archive.exists() or archive.stat().st_size==0:
        tmp=archive.with_suffix('.part')
        for attempt in range(4):
            try:
                offset=tmp.stat().st_size if tmp.exists() else 0
                request=urllib.request.Request(SOURCE+name+'_Mulan.zip',headers={'Range':f'bytes={offset}-'} if offset else {})
                with urllib.request.urlopen(request,timeout=90) as response:
                    append=response.status==206 and offset>0
                    with tmp.open('ab' if append else 'wb') as target:
                        while chunk:=response.read(64*1024): target.write(chunk)
                break
            except (TimeoutError,OSError):
                if attempt==3: raise
                time.sleep(1)
        # Inspect before moving. Never extract paths from an untrusted archive.
        with zipfile.ZipFile(tmp) as z:
            if not any(p.endswith('.arff') for p in z.namelist()): raise ValueError('Missing ARFF')
        tmp.replace(archive)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    if parsed.exists():
        with np.load(parsed,allow_pickle=False) as z:
            if str(z['sha'])==digest:
                return z['x'],z['y'],json.loads(str(z['metadata']))
    with zipfile.ZipFile(archive) as z:
        xml=next(p for p in z.namelist() if p.lower().endswith('.xml'))
        arff=next(p for p in z.namelist() if p.lower().endswith('.arff'))
        labels=[e.attrib['name'] for e in ET.fromstring(z.read(xml)).iter() if e.tag.split('}')[-1]=='label']
        x,y,features=parse_arff(z.read(arff).decode('utf-8-sig'),labels)
    # FNR is defined here on annotated examples with at least one positive label.
    keep=y.sum(1)>0
    metadata=dict(name=name,url=SOURCE+name+'_Mulan.zip',sha256=digest,rows=len(x),
                  excluded_unannotated=int((~keep).sum()),features=features,labels=labels)
    x,y=x[keep],y[keep]
    np.savez_compressed(parsed,x=x,y=y,sha=digest,metadata=json.dumps(metadata))
    return x,y,metadata


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--datasets',nargs='+',default=DATASETS); a=p.parse_args()
    for name in a.datasets:
        x,y,m=load(name); print(name,x.shape,y.shape,'excluded',m['excluded_unannotated'],flush=True)
