# This file contains two functions for generating synthetic data for testing
# the imputation methods.
# Copyright (C) 2024 Georgios Is. Detorakis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numpy as np

rng = np.random.default_rng()


def generateSyntheticData(size=1000, missing_perc=0.2):
    """! Generates random data points drawn from a normal distribution with
    zero mean and unit standard deviation.

    @param size The number of data points to be generated (int).
    @param missing_perc The percentage (0 < p < 1) of the elements that will
    receive a NaN value (missing values) (float).

    @return A tuple of two ndarrays of shape (size, ). The first contains NaNs,
    and the second is the intact array.
    """
    n_missing_vals = int(missing_perc * size)
    data = rng.standard_normal(size)
    data0 = data.copy()

    index = np.array([i for i in range(size)], dtype='i')
    missing_idxs = rng.choice(index, size=n_missing_vals)
    data[missing_idxs] = np.nan
    return data, data0


def generateSinusoidalData(size=1000, missing_perc=0.2, freq=5):
    """! This function generates sinusoidal data points and adds some noise
    drawn from a normal distributions with zero mean and standard deviation
    of 0.05.

    @param size Size of new generated array (int).
    @param missing_perc The percentage (0 < p < 1) of the elements that will
    receive a NaN value (missing values) (float).
    @frew The frequency of the sinusoidal signal (float).

    @return A tuple of two ndarrays of shape (size, ). The first contains NaNs,
    and the second is the intact array.
    """
    n_missing_vals = int(missing_perc * size)
    t = np.linspace(-2.*np.pi, 2.*np.pi, size)
    data = np.sin(2.0*np.pi*freq*t)
    data += np.random.normal(0, 0.05, size)
    data0 = data.copy()

    index = np.array([i for i in range(size)], dtype='i')
    missing_idxs = rng.choice(index, size=n_missing_vals)
    data[missing_idxs] = np.nan
    return data, data0
