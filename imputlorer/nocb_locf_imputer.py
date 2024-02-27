# This fils contains the class that implements the NOCB and LOCF imputers.
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


class NOCBLOCFImputer:
    """! This class implements a NOCB (Next Observation Carried Backward)
    method, similar to Pandas bfill() method, and an LOCF (Last Observation
    Carried Forward), similar to Pandas fill () method.
    """
    def __init__(self,
                 strategy="locf",   # or nocb
                 missing_values=np.nan,
                 copy=True):
        """!
        Constructor of NOCBLOCFImputer class.

        @param strategy Its either "locf" or "nocb" (str).
        @param missing_values The value of missing values (default is NaN).
        @param copy It True, it copies the input array (bool) and operates on a
        copied array.

        @return void
        """
        self.strategy = strategy
        self.strategy_cp = strategy
        self.missing_values = missing_values
        self.copy = copy
        self.correct_nans = False

    def fit_transform(self, X):
        """! Applies the NOCF or NOCB imputation methods on the input array X.

        @param X Input ndarray of shape (n, 1).

        @note This method takes care of the boundary values in contrast with
        the Pandas method, leaving the boundary untouched. This means if NaN
        values are at the array's endpoints, they will be replaced by the first
        or last non-nan value depending on the chosen strategy. To achieve that,
        it runs one round of NOCB or LOCF and then runs more rounds to fill
        the missing values depending on which endpoint there are NaN values.

        @return A ndarray of shape (n, 1) that contains the imputed data.
        """
        x = self.transform(X)
        if np.isnan(x[0, :]).sum() > 0:
            if np.isnan(x[0, 0]) and self.strategy == "locf":
                self.strategy = "nocb"
                x = self.transform(x)
            if np.isnan(x[0, -1]) and self.strategy == "locf":
                self.stategy = "locf"
                x = self.transform(x)
            if np.isnan(x[0, 0]) and self.strategy == "nocb":
                self.strategy = "nocb"
                x = self.transform(x)
            if np.isnan(x[0, -1]) and self.strategy == "nocb":
                self.correct_nans = True
                self.stategy = "locf"
                x = self.transform(x)
            self.reset()
        x = x.T
        return x

    def reset(self):
        """! It resets the `strategy` to its value chosen by the user and the
        `correct_nans` to `False`.
        """
        self.strategy = self.strategy_cp
        self.correct_nans = False

    def transform(self, X):
        """! Implements the core transformation of the NOCB and LOCF methods.
        It returns a ndarray that contains the imputed values.

        @param X The input ndarray of shape (n, 1).

        @return A ndarray of shape (n, 1).
        """
        if self.copy:
            x = X.copy()
        else:
            x = X

        if self.strategy == "nocb" and self.correct_nans is False:
            x = np.flip(x)
        # FIXME Make the following statement more generic (include None, 0,
        # etc)
        mask_ = np.isnan(x)
        index_ = np.where(~mask_,
                          np.arange(mask_.shape[1]), 0)
        np.maximum.accumulate(index_, axis=1, out=index_)
        x[mask_] = x[np.nonzero(mask_)[0], index_[mask_]]

        if self.strategy == "nocb" and self.correct_nans is False:
            x = np.flip(x)
        return x
