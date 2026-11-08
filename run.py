import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import torch
from data import boards, source_pools, corrupt
from train import variants
from model import probabilities
from risk import crc_threshold, corc_threshold, choose_t, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 23, 42, 71, 101, 131, 173, 211, 257, 307])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--out', default='results/raw')
    a = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ntrain, nval, ncal, ntest = (256, 64, 100, 100) if a.smoke else (6000, 1000, 1000, 1500)
    warm_epochs, fine_epochs = (2, 1) if a.smoke else (35, 20)
    rows, histories, demo = [], {}, {}
    start = time.perf_counter()
    for seed in a.seeds:
        t0 = time.perf_counter()
        images, labels, pool, heldout = source_pools(seed)
        assert not set(pool) & set(heldout)
        x, y = boards(images, labels, pool, ntrain, seed)
        xv, yv = boards(images, labels, heldout, nval, seed+1000)
        xc, yc = boards(images, labels, heldout, ncal, seed+2000)
        xt, yt = boards(images, labels, heldout, ntest, seed+3000)
        models = variants(x, y, xv, yv, seed, warm_epochs, fine_epochs)
        predictions = {}
        for name, (model, history) in models.items():
            histories[f'{seed}/{name}'] = history
            pc, pv = probabilities(model, xc), probabilities(model, xv)
            predictions[name] = dict(cal=pc.tolist(), validation=pv.tolist(), test={})
            for condition in ['clean', 'noise', 'occlusion']:
                shifted = corrupt(xt, condition, seed+4000)
                pt = probabilities(model, shifted)
                predictions[name]['test'][condition] = pt[:96].tolist()
                for alpha in [0.05, 0.1, 0.2]:
                    t = choose_t(pv, yv, alpha, 0.8)
                    # Fresh labeled calibration can restore exchangeability after a known shift.
                    fresh_cal = probabilities(model, corrupt(xc, condition, seed+5000))
                    fresh_val = probabilities(model, corrupt(xv, condition, seed+6000))
                    fresh_t = choose_t(fresh_val, yv, alpha, 0.8)
                    for control, q in [('fixed', 0.5), ('crc', crc_threshold(pc, yc, alpha)),
                                       ('cvar', corc_threshold(pc, yc, alpha, 0.8, t)),
                                       ('crc_fresh', crc_threshold(fresh_cal,yc,alpha)),
                                       ('cvar_fresh', corc_threshold(fresh_cal,yc,alpha,0.8,fresh_t)),
                                       ('crc_small', crc_threshold(pc[:100],yc[:100],alpha))]:
                        rows.append(dict(seed=seed, model=name, condition=condition,
                                         alpha=alpha, control=control, threshold=q,
                                         t=fresh_t if control=='cvar_fresh' else t if control=='cvar' else None,
                                         **metrics(pt, yt, q)))
        if seed == a.seeds[0]:
            demo = dict(seed=seed, cal_y=yc.tolist(), val_y=yv.tolist(), y=yt[:96].tolist(),
                        images={k: np.round(corrupt(xt, k, seed+4000)[:96]*255).astype(int).tolist()
                                for k in ['clean', 'noise', 'occlusion']}, models=predictions)
            for name, (model, _) in models.items():
                torch.save(model.state_dict(), out / f'{name}.pt')
        print(f'seed {seed}: {time.perf_counter()-t0:.1f}s', flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(out/'metrics.csv', index=False)
    frame.groupby(['model','condition','alpha','control'])[['fnr','fpr','cvar80','size','exact']].agg(['mean','std']).to_csv(out/'summary.csv')
    manifest = dict(seeds=a.seeds, sizes=dict(train=ntrain,validation=nval,calibration=ncal,test=ntest),
                    epochs=dict(warm=warm_epochs,fine=fine_epochs), seconds=time.perf_counter()-start,
                    python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    sklearn=sklearn.__version__, threads=4, smoke=a.smoke,
                    source_sha256=hashlib.sha256(images.tobytes()+labels.tobytes()).hexdigest())
    (out/'run.json').write_text(json.dumps(manifest, indent=2))
    (out/'history.json').write_text(json.dumps(histories))
    (out/'demo.json').write_text(json.dumps(demo, separators=(',',':')))
    hashes={name:hashlib.sha256((out/f'{name}.pt').read_bytes()).hexdigest() for name in models}
    (out/'checkpoint_hashes.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
