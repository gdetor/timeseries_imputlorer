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

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
# sys.pycache_prefix = os.path.join(root_dir, 'dist', 'pycache', 'test')
sys.pycache_prefix = os.path.join(root_dir, '__pycache__', 'test')
lore = __import__('imputlorer')


X = np.array([1, 2, 3, np.nan, 5, 6, 7, np.nan]).reshape(-1, 1)
X_ = np.array([1, 2, 3, 5, 6, 7]).reshape(-1, 1)
mu = X_.mean()
X_mean = np.array([1, 2, 3, mu, 5, 6, 7, mu]).reshape(-1, 1)
median = np.median(X_)
X_median = np.array([1, 2, 3, median, 5, 6, 7, median]).reshape(-1, 1)
X_const = np.array([1, 2, 3, 3, 5, 6, 7, 3]).reshape(-1, 1)
X_locf = np.array([1, 2, 3, 3, 5, 6, 7, 7]).reshape(-1, 1)
X_nocb = np.array([1, 2, 3, 5, 5, 6, 7, 7]).reshape(-1, 1)


class TestClassSimpleMethod:
    def test_simple_mean(self):
        imp = lore.imputer.TSImputer(method="simple", strategy="mean")
        Y = imp.run(X)
        assert np.array_equal(Y, X_mean)

    def test_simple_median(self):
        imp = lore.imputer.TSImputer(method="simple", strategy="median")
        Y = imp.run(X)
        assert np.array_equal(Y, X_median)

    def test_simple_constant(self):
        imp = lore.imputer.TSImputer(method="simple",
                                     strategy="constant",
                                     fill_value=3)
        Y = imp.run(X)
        assert np.array_equal(Y, X_const)


class TestClassIterativeMethod:
    def test_simple_mean(self):
        imp = lore.imputer.TSImputer(method="multi", strategy="mean")
        Y = imp.run(X)
        assert np.array_equal(Y, X_mean)

    def test_simple_median(self):
        imp = lore.imputer.TSImputer(method="multi", strategy="median")
        Y = imp.run(X)
        assert np.array_equal(Y, X_median)

    def test_simple_constant(self):
        imp = lore.imputer.TSImputer(method="multi",
                                     strategy="constant",
                                     fill_value=3)
        Y = imp.run(X)
        assert np.array_equal(Y, X_const)


class TestClassKNNMethod:
    def test_knn(self):
        imp = lore.imputer.TSImputer(method="knn",
                                     n_neighbors=5)
        Y = imp.run(X)
        assert np.array_equal(Y, X_mean)


class TestClassNOCB:
    def test_nocb(self):
        imp = lore.imputer.TSImputer(method="nocb")
        Y = imp.run(X)
        assert np.array_equal(Y[:-1], X_nocb[:-1])


class TestClassLOCF:
    def test_locf(self):
        imp = lore.imputer.TSImputer(method="locf")
        Y = imp.run(X)
        assert np.array_equal(Y, X_locf)
