import numpy as np

rng = np.random.default_rng()


def generateSyntheticData(size=1000, missing_perc=0.2):
    n_missing_vals = int(missing_perc * size)
    data = rng.standard_normal(size)
    missing_idxs = rng.integers(low=0, high=size, size=n_missing_vals)
    data[missing_idxs] = np.nan
    return data
