"""Finite-sample calibration for per-image missed-label fractions."""
import numpy as np
import torch


def validate(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if p.shape != y.shape or p.ndim != 2 or len(p) == 0:
        raise ValueError('Expected nonempty equal-shaped probability and label matrices')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be finite and in [0, 1]')
    if not np.isin(y, [0, 1]).all() or np.any(y.sum(1) == 0):
        raise ValueError('Each example needs at least one binary positive label')
    return p, y


def losses(p, y, threshold):
    return ((p < threshold) * y).sum(1) / y.sum(1)


def crc_threshold(p, y, alpha):
    """Largest feasible breakpoint; >= includes ties at the boundary.

    Each image has total weight one, regardless of its label count.
    The +1 correction is a bound on one unseen image's loss.
    """
    p, y = validate(p, y)
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0, 1)')
    budget = (len(p) + 1) * alpha - 1
    if budget < 0:
        return 0.0
    positive = y.astype(bool)
    scores = p[positive]
    weights = np.broadcast_to(1 / y.sum(1, keepdims=True), y.shape)[positive]
    values, inverse = np.unique(scores, return_inverse=True)
    cumulative = np.cumsum(np.bincount(inverse, weights=weights))
    blocked = np.flatnonzero(cumulative > budget + 1e-12)
    return float(values[blocked[0]]) if len(blocked) else 1.0


def differentiable_threshold(p, y, alpha):
    """Exact piecewise derivative through the active positive score (CRT).

    Sorting/index selection is discrete; the chosen score keeps its gradient.
    At ties this selects one deterministic subgradient, not a unique derivative.
    """
    budget = (len(p) + 1) * alpha - 1
    if budget < 0:
        return p.sum() * 0
    mask = y.bool()
    values, order = torch.sort(p[mask])
    weights = (1 / y.sum(1, keepdim=True)).expand_as(y)[mask][order]
    # Select by the cumulative loss after removing a whole group of ties.
    unique, counts = torch.unique_consecutive(values.detach(), return_counts=True)
    ends = counts.cumsum(0) - 1
    blocked = torch.nonzero(weights.cumsum(0)[ends] > budget + 1e-7)
    if not len(blocked):
        return p.sum() * 0 + 1
    group = int(blocked[0, 0])
    start = 0 if group == 0 else int(ends[group - 1]) + 1
    return values[start]


def corc_threshold(p, y, alpha, delta, t):
    """CVaR calibration with t fixed independently of this calibration set."""
    p, y = validate(p, y)
    if not 0 < alpha < 1 or not 0 <= delta < 1 or not 0 <= t <= alpha:
        raise ValueError('Invalid CVaR parameters')
    upper = t + (1 - t) / (1 - delta)
    def criterion(q):
        transformed = t + np.maximum(losses(p, y, q) - t, 0) / (1 - delta)
        return (transformed.sum() + upper) / (len(p) + 1)
    if criterion(0) > alpha:
        return 0.0  # all labels included: risk is identically zero
    lo, hi = 0.0, 1.0
    for _ in range(45):
        mid = (lo + hi) / 2
        if criterion(mid) <= alpha:
            lo = mid
        else:
            hi = mid
    return lo


def choose_t(p, y, alpha, delta):
    grid = np.linspace(0, alpha, 9)
    return float(max(grid, key=lambda t: corc_threshold(p, y, alpha, delta, t)))


def empirical_cvar(values, delta=0.8):
    """Fractional tail mass handles atoms and n*(1-delta) non-integers."""
    v = np.sort(np.asarray(values, float))[::-1]
    mass = len(v) * (1 - delta)
    whole = int(np.floor(mass))
    total = v[:whole].sum()
    if whole < len(v):
        total += (mass - whole) * v[whole]
    return float(total / mass)


def metrics(p, y, threshold):
    pred = p >= threshold
    fnr = losses(p, y, threshold)
    fp = ((1 - y) * pred).sum(1) / (1 - y).sum(1)
    return dict(fnr=float(fnr.mean()), fpr=float(fp.mean()),
                cvar80=empirical_cvar(fnr), size=float(pred.sum(1).mean()),
                exact=float((pred == y).all(1).mean()))
