"""Multi-label boards assembled from the bundled UCI handwritten digits."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def source_pools(seed):
    d = load_digits()
    train, heldout = train_test_split(np.arange(len(d.target)), test_size=0.4,
                                     stratify=d.target, random_state=seed)
    return d.images.astype('float32') / 16, d.target, train, heldout


def boards(images, labels, pool, n, seed):
    rng = np.random.default_rng(seed)
    x, y = np.zeros((n, 20, 20), 'float32'), np.zeros((n, 10), 'float32')
    classes = [pool[labels[pool] == j] for j in range(10)]
    for i in range(n):
        chosen = rng.choice(10, size=int(rng.integers(1, 4)), replace=False)
        slots = rng.choice(4, size=len(chosen), replace=False)
        for digit, slot in zip(chosen, slots):
            image = images[rng.choice(classes[digit])]
            a, b = (slot // 2) * 10 + 1, (slot % 2) * 10 + 1
            x[i, a:a+8, b:b+8] = image * rng.uniform(0.75, 1)
        y[i, chosen] = 1
    return x.reshape(n, -1), y


def corrupt(x, kind, seed):
    rng = np.random.default_rng(seed)
    v = x.reshape(-1, 20, 20).copy()
    if kind == 'noise':
        v = np.clip(v * 0.55 + rng.normal(0, 0.18, v.shape), 0, 1)
    elif kind == 'occlusion':
        for image in v:
            a, b = rng.integers(0, 15, 2)
            image[a:a+6, b:b+6] = 0
    elif kind != 'clean':
        raise ValueError(kind)
    return v.reshape(len(v), -1).astype('float32')
