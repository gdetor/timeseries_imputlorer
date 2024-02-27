# This script contains all the test cases for the TSImputer class.
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
import os
import sys

import pytest
import numpy as np
from scipy import stats

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
# sys.pycache_prefix = os.path.join(root_dir, 'dist', 'pycache', 'test')
sys.pycache_prefix = os.path.join(root_dir, '__pycache__', 'test')
lore = __import__('imputlorer')

stat_dict = {}
X = np.array([1, 2, 3, 4, 5, 6, 7], dtype='f').reshape(-1, 1)
Y = np.array([1, 2, 3, 4, 5, 6, 7], dtype='f').reshape(-1, 1)
stat_dict["count"] = len(X)
stat_dict["n_missing_values"] = np.isnan(X).sum()

stat_dict["min"] = X.min()
stat_dict["max"] = X.max()
mu = np.mean(X)
stat_dict["mean"] = mu
sd = np.std(X)
stat_dict["std"] = sd
med = np.median(X)
stat_dict["median"] = med
mod = stats.mode(X)
stat_dict["mode"] = mod
per = np.percentile(X, [25, 50, 75])
stat_dict["perc_25"] = per[0]
stat_dict["perc_50"] = per[1]
stat_dict["perc_75"] = per[2]
kurtosis = stats.kurtosis(X, axis=0, bias=True)
stat_dict["kurtosis"] = kurtosis
skewness = stats.skew(X, axis=0, bias=True)
stat_dict["skewness"] = skewness
CI = stats.t.interval(confidence=0.9,
                      df=len(X)-1,
                      loc=mu,
                      scale=stats.sem(X))
stat_dict["CI"] = CI


class TestClassSummaryStatistics:
    def test_summary(self):
        stat = lore.compare_imputation_methods.summaryStatistics(X)
        assert stat_dict["count"] == stat["count"]
        assert stat_dict["n_missing_values"] == stat["n_missing_values"]
        assert np.isclose(stat_dict["min"], stat["min"])
        assert np.isclose(stat_dict["max"], stat["max"])
        assert np.isclose(stat_dict["mean"], stat["mean"])
        assert np.isclose(stat_dict["std"], stat["std"])
        assert np.isclose(stat_dict["median"], stat["median"])
        assert np.isclose(stat_dict["mode"].mode, stat["mode"].mode)
        assert np.isclose(stat_dict["perc_25"], stat["perc_25"])
        assert np.isclose(stat_dict["perc_50"], stat["perc_50"])
        assert np.isclose(stat_dict["perc_75"], stat["perc_75"])
        assert np.isclose(stat_dict["skewness"], stat["skewness"])
        assert np.isclose(stat_dict["kurtosis"], stat["kurtosis"])
        assert np.isclose(stat_dict["CI"][0], stat["CI"][0])
        assert np.isclose(stat_dict["CI"][1], stat["CI"][1])


class TestClassBias:
    def test_raw_bias(self):
        err = lore.compare_imputation_methods.RB(X, Y)
        assert err == 0.0

    def test_percent_bias(self):
        err = lore.compare_imputation_methods.PB(X, Y)
        assert err == 0.0


class TestClassCompareImputationMethods:
    def test_methods(self):
        methods = ["simple", "multi", "knn"]
        standalone_methods = ["nocb", "locf"]
        strategy = ["mean", "median", "most_frequent", "constant"]
        err = lore.compare_imputation_methods.compareImputationMethods(
                X,
                method=methods,
                standalone_method=standalone_methods,
                strategy=strategy,
                missing_vals_perc=0,
                print_on=False
                )
        for _, item in err.items():
            assert item == [0.0, 0.0, 0.0, 0.0, 1.0]
