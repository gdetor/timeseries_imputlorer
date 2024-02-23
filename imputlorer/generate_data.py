import numpy as np

rng = np.random.default_rng()


def generateSyntheticData(size=1000, missing_perc=0.2):
    n_missing_vals = int(missing_perc * size)
    data = rng.standard_normal(size)
    data0 = data.copy()

    index = np.array([i for i in range(size)], dtype='i')
    missing_idxs = rng.choice(index, size=n_missing_vals)
    data[missing_idxs] = np.nan
    return data, data0


def generateSinusoidalData(size=1000, missing_perc=0.2, freq=5):
    n_missing_vals = int(missing_perc * size)
    t = np.linspace(-2.*np.pi, 2.*np.pi, size)
    data = np.sin(2.0*np.pi*freq*t)
    data += np.random.normal(0, 0.05, size)
    data0 = data.copy()

    index = np.array([i for i in range(size)], dtype='i')
    missing_idxs = rng.choice(index, size=n_missing_vals)
    data[missing_idxs] = np.nan
    return data, data0
