import numpy as np

def kernel_target_alignment(kernel, y):
    K = np.asarray(kernel, dtype=float)
    y = np.asarray(y, dtype=float)
    if set(np.unique(y)) <= {0.0, 1.0}:
        y = np.where(y == 0.0, -1.0, 1.0)
    yy = np.outer(y, y)
    numerator = np.sum(K * yy)
    denominator = np.sqrt(np.sum(K * K) * np.sum(yy * yy))
    return float(numerator / denominator) if denominator else 0.0
