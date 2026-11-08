"""One warm start, then matched training budgets for four fine-tuning rules."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from model import Recognizer
from risk import differentiable_threshold


def fit(model, x, y, xv, yv, epochs, seed, mode='bce', alpha=0.1):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, y = torch.as_tensor(x), torch.as_tensor(y)
    xv, yv = torch.as_tensor(xv), torch.as_tensor(yv)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    best, best_loss, log = None, float('inf'), []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x))
        costs = []
        for start in range(0, len(x), 256):
            idx = order[start:start+256]
            if len(idx) < 32:
                continue
            logits = model(x[idx]); target = y[idx]
            if mode in ('crt', 'stopgrad'):
                p = logits.sigmoid(); half = len(idx) // 2
                q = differentiable_threshold(p[:half], target[:half], alpha)
                if mode == 'stopgrad':
                    q = q.detach()
                soft = torch.sigmoid((p[half:] - q) / 0.05)
                false_positive = (soft * (1-target[half:])).sum(1) / (1-target[half:]).sum(1)
                cost = false_positive.mean() + 0.3 * F.binary_cross_entropy_with_logits(logits, target)
            else:
                weight = torch.full((10,), 3.0) if mode == 'weighted' else None
                cost = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
            opt.zero_grad(); cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step(); costs.append(float(cost.detach()))
        model.eval()
        with torch.no_grad():
            # Shared selection rule; final calibration labels never choose a checkpoint.
            val = float(F.binary_cross_entropy_with_logits(model(xv), yv))
        log.append(dict(epoch=epoch+1, objective=float(np.mean(costs)), validation_bce=val))
        if val < best_loss:
            best_loss, best = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model, log


def variants(x, y, xv, yv, seed, warm_epochs, fine_epochs):
    torch.manual_seed(seed)
    warm, log = fit(Recognizer(), x, y, xv, yv, warm_epochs, seed)
    result = {'warm': (copy.deepcopy(warm), log)}
    for name in ['bce', 'weighted', 'crt', 'stopgrad']:
        result[name] = fit(copy.deepcopy(warm), x, y, xv, yv, fine_epochs, seed+1, name)
    return result
