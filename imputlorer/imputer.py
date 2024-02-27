# TSImputer class implementation. This is the main class that implements all
# the necessary methods for univariate time series data imputation.
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

from sklearn.linear_model import BayesianRidge, Ridge, LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer

from .nocb_locf_imputer import NOCBLOCFImputer


class TSImputer:
    """! This class implements all the necessary methods for univariate time
    series data imputation.
    """
    def __init__(self,
                 method="simple",
                 strategy="mean",
                 missing_values=np.nan,
                 fill_value=0.0,
                 estimator='bayesian',
                 sample_posterior=False,
                 max_iter=10,
                 tol=1e-3,
                 n_nearest_features=None,
                 imputation_order="ascending",
                 skip_complete=False,
                 min_value=-np.inf,
                 max_value=np.inf,
                 random_state=None,
                 n_neighbors=5,
                 weights="uniform",
                 metric="nan_euclidean",
                 ):
        """!
        The constructor of TSImputer class determines which data imputation
        method and what strategy will be used. In addition, sets all the
        parameters provided by the user.

        @param method It determines which imputation method will be used.
        `nocb` for Next Observation Carried Backwards, `locf` for Last
        Observation Carried Forward, `simple` for engaging the `SimpleImputer`
        provided by the Sklearn package, `multi` for invoking the
        IterativeImputer of Sklearn, and finally `knn` for the KNNImputer of
        Sklearn (str).
        @param strategy Determines the value that will replace the missing
        ones. It can be one of `mean`, `median`, `constant`, or
        `most_frequent` (str).
        @param missing_values Defines the value of the missing values (by
        default NaN) (float).
        @param fill_value It is the value that will replace the missing ones in
        case the strategy `constant` has been chosen (float)
        @param estimator The estimator that will be used by the
        IterativeImputer. It can be one of `bayesian` (default), `linear`, or
        `ridge`.
        @param sample_posterion Whether to sample from the (Gaussian)
        predictive posterior of the fitted estimator for each imputation, when
        the IterativeImputer has been selected (bool).
        @param max_iter Maximum number of imputation rounds to perform before
        returning the imputations computed during the final round (int).
        @param tol Tolerance of the stopping condition (float).
        @param n_nearest_features Number of other features to use to estimate
        the missing values of each feature column (int).
        @param imputation_order The order in which the features will be
        imputed. The user can choose between: 'ascending': From features with
        fewest missing values to most. 'descending': From features with most
        missing values to fewest. 'roman': Left to right. 'arabic': Right to
        left. 'random': A random order for each round (str).
        @param skip_complete If True then features with missing values during
        transform which did not have any missing values during fit will be
        imputed with the initial imputation method only (bool).
        @param min_value Minimum possible imputed value (float).
        @param max_value Maximum possible imputed value (float).
        @param random_state The seed of the pseudo random number generator to
        use (int).
        @param n_neighbors Number of neighboring samples to use for
        imputation (int). To use with KNNImputer.
        @param weights Weight function used in prediction. The user can choose
        one of the following: 'uniform' : uniform weights. All points in each
        neighborhood are weighted equally. 'distance': weight points by the
        inverse of their distance. callable: a user-defined function which
        accepts an array of distances, and returns an array of the same shape
        containing the weights.
        @param metric Distance metric for searching neighbors. The user can
        choose one of the following: 'nan_euclidean' or callable: a
        user-defined function which conforms to the definition of
        _pairwise_callable(X, Y, metric, **kwds). The function accepts two
        arrays, X and Y, and a missing_values keyword in kwds and returns a
        scalar distance value.

        @note For more details on the parameters, the user is refered to the
        sklearn documentation page:
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute

        @return void
        """
        self.missing_values = missing_values
        self.method = method

        # Choose one method and pass the corresponding parameters to its
        # constructor
        if method == "nocb":
            self.imputer = NOCBLOCFImputer(strategy=method,
                                           missing_values=missing_values,
                                           )
        elif method == "locf":
            self.imputer = NOCBLOCFImputer(strategy=method,
                                           missing_values=missing_values,
                                           )
        elif method == "simple":
            self.imputer = SimpleImputer(missing_values=missing_values,
                                         strategy=strategy,
                                         fill_value=fill_value
                                         )
        elif method == "multi":
            if estimator == "bayesian":
                estimator = BayesianRidge()
            elif estimator == "linear":
                estimator = LinearRegression()
            else:
                estimator = Ridge()
            self.imputer = IterativeImputer(missing_values=missing_values,
                                            initial_strategy=strategy,
                                            fill_value=fill_value,
                                            estimator=estimator,
                                            sample_posterior=sample_posterior,
                                            max_iter=max_iter,
                                            tol=tol,
                                            n_nearest_features=n_nearest_features,
                                            imputation_order=imputation_order,
                                            skip_complete=skip_complete,
                                            min_value=min_value,
                                            max_value=max_value,
                                            random_state=random_state
                                            )
        elif method == "knn":
            self.imputer = KNNImputer(missing_values=missing_values,
                                      n_neighbors=n_neighbors,
                                      weights=weights,
                                      metric=metric
                                      )
        else:
            print("Not a valid method!")
            print("Choose simple, multi, or knn!")
            exit()

    def run(self, X):
        """! This method applies a fit_transform. First, it checks the
        dimensions of the input array and reshapes it in case its shape is
        (n,). Then, if the chosen imputation method is "locf" or "nocb", it
        transposes the input array. Finally, it applies the fit_transform and
        returns the results.

        @param X Input array of shape (n, ) or (n, 1) (ndarray).

        @return A new numpy array where the missing values (NaNs in X) have
        been imputed.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if self.method == "locf" or self.method == "nocb":
            X = X.T
        X_imputed = self.imputer.fit_transform(X)
        return X_imputed

    def getDataRange(self, x):
        """! Returns the range of the values in the input numpy array x,
        ignoring any missing values (NaNs).

        @param x Input array of shape (n, ) or (n, 1) (ndarray).

        @return A Python tuple that contains the minimum and the maximum values.
        """
        return np.nanmin(x), np.nanmax(x)

    def getDataMidRange(self, x):
        """! Returns the midrange of the values in the input array x, ignoring
        any missing values NaNs.

        @param x Input array of shape (n, ) or (n, 1) (ndarray).

        @return The midrange as a float value.
        """
        x_min, x_max = self.getDataRange(x)
        return (x_max + x_min) / 2

    def getStatistics(self):
        """! Returns the imputation fill value for each feature. It discards
        statistics related to NaNs.

        @return An ndarray of shape (n,)
        """
        if self.method == "simple":
            return self.imputer.statistics_
        else:
            print("No statistics available for NOCB or LOCF methods!")
            return -1

    def getIndicator(self):
        """! Returns the indicator used to add binary indicators for missing
        values.

        @return An indicator of type `MissingIndicator` (see sklearn doc).
        """
        return self.imputer.indicator_

    def getNumFeatures(self):
        """! Returns the number of features seen during fit.

        @return Number of features as an integer.
        """
        return self.imputer.n_features_in_

    def getFeaturesNames(self):
        """! Returns the names of features seen during fit. Defined only when
        the input array has feature names that are all strings.

        @return An ndarray of shape (n_features_in, ) that contains all
        features' names.
        """
        return self.imputer.feature_names_in_
