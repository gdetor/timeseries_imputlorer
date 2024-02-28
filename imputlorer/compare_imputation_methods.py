# This script implements functions for summary statistics and metrics for
# comparing different imputation methods.
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
from scipy import stats

from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

from imputlorer.imputer import TSImputer


def summaryStatistics(X, report=False):
    """! This function computes and prints the summary statistics of an array
    of shape (n_samples, 1) at the standard output. In addition, it estimates
    the skewness, kurtosis, and the confidence interval.

    @param X An ndarray of shape (n_samples, 1).

    @return A Python dictionary that contains the summary statistics.
    """
    X_ = X[~np.isnan(X)]

    stat_dict = {}
    # Mean
    mu = np.mean(X_)
    stat_dict["mean"] = mu

    # Standard deviation
    sd = np.std(X_)
    stat_dict["std"] = sd

    # Median
    med = np.median(X_)
    stat_dict["median"] = med

    # Mode
    mod = stats.mode(X_)
    stat_dict["mode"] = mod

    # Percentiles
    per = np.percentile(X_, [25, 50, 75])
    stat_dict["perc_25"] = per[0]
    stat_dict["perc_50"] = per[1]
    stat_dict["perc_75"] = per[2]

    # Count missing values
    missing_vals = np.isnan(X).sum()
    stat_dict["n_missing_values"] = missing_vals

    # Count elements of X
    stat_dict["count"] = len(X)

    # Minimum and maximum (range of X)
    stat_dict["min"] = X.min()
    stat_dict["max"] = X.max()

    # Compute kurtosis and skewness
    kyrtosis = stats.kurtosis(X_, axis=0, bias=True)
    skewness = stats.skew(X_, axis=0, bias=True)
    stat_dict["kurtosis"] = kyrtosis
    stat_dict["skewness"] = skewness

    # Estimate the confidence interval
    CI = stats.t.interval(confidence=0.9,
                          df=len(X_)-1,
                          loc=mu,
                          scale=stats.sem(X_))
    stat_dict["CI"] = CI

    if report:
        print("===============================")
        print("Summary statistics")
        print(f"Missing values  (count): {missing_vals}")
        print(f"Count: {len(X)}")
        print(f"Min: {X_.min()}")
        print(f"Mean: {mu}")
        print(f"SD: {sd}")
        print(f"Median: {med}")
        print(f"Mode: {mod}")
        print(f"25%: {per[0]}")
        print(f"50%: {per[1]}")
        print(f"75%: {per[2]}")
        print(f"Max: {X_.max()}")
        print("-------------------------------")
        print(f"Kurtosis: {kyrtosis} and skewness: {skewness}")
        print(f"Confidence Interval: {CI}")
        print("===============================")

    return stat_dict


def RB(y_true, y_pred):
    """! Estimates the raw bias as the mean of the difference between true
    (target) and predicted values.

    @param y_true Ground truth (target) values, ndarray of shape (n_samples, ).
    @param y_pred Estimated target (predicted) values, ndarray of shape
    (n_samples,).

    @note Raw bias by definition has to be close to zero, when y_pred and
    y_true are close.

    @return Raw bias as a float value.
    """
    return np.mean(y_true - y_pred)


def PB(y_true, y_pred):
    """! Estimates the percent bias as the mean of the difference between true
    (target) and predicted values.

    @param y_true Ground truth (target) values, ndarray of shape (n_samples, ).
    @param y_pred Estimated target (predicted) values, ndarray of shape
    (n_samples, ).

    @note A percent bias < 5% is acceptable.

    @return Percent bias as a float value.
    """
    return 100.0 * np.mean((y_true - y_pred) / np.abs(y_true))


def compareImputationMethods(X,
                             method=["simple"],
                             standalone_method=["nocb"],
                             strategy=["mean"],
                             missing_vals_perc=0.2,
                             print_on=True):
    """! It compares several imputation methods and strategies using five
    metrics: Mean absolute error, root mean squared error, R2 score, raw bias,
    and percent bias. The user can use this function to assess which imputation
    method is the most suitable for their data. The user must pass the raw data
    and choose the method, strategy, or strategies. The function reports all
    the errors between the imputed data time series and the intact data.

    @attention The data in X must contain non-nan values. This function will
    replace X's elements with NaNs at random.

    @param X ndarray of shape (n_samples, 1) without any NaNs.
    @param method Imputation method: "simple" - Sklearn Simple Imputer,
    "multi" - Sklearn IterativeImputer, "knn" - Sklearn KNNImputer (list). More
    than one method can be used.
    @param standalone_method Imputation methods without specific strategy:
    "nocb" next observation carried backwards, "locf" - last observation
    carried forward (list). One or both methods can be used.
    @param strategy Imputation strategy. This can be "mean", "median",
    "constant", or "most_frequent", or a combination of those or all together
    (list).
    @param missing_vals_perc The percentage of X's elements that NaNs will
    replaced (0.0 < percentage < 1.0) (float).
    @param print_on If True, it prints the metrics for each method at the
    standard output.

    @return A Python dictionary that contains the name of the method and the
    corresponding error metrics.
    """
    # Determine the number of missing values
    n_missing_vals = int(missing_vals_perc * len(X))

    # Make a copy of the input array
    pre_imputed_vals = X.copy()

    # Create the missing values mask
    if n_missing_vals != 0:
        missing_vals_mask = np.random.choice([i for i in range(len(X))],
                                             size=n_missing_vals)
        # Replace the candidate values with NaN
        X[missing_vals_mask] = np.nan

    # Impute the data and store the results in a dictionary
    res = {}
    for i, m in enumerate(method):
        for j, s in enumerate(strategy):
            imp = TSImputer(method=m, strategy=s)
            X_imp = imp.run(X)
            res[m+"-"+s] = X_imp

    if len(standalone_method) != 0:
        for i, m in enumerate(standalone_method):
            imp = TSImputer(method=m)
            X_imp = imp.run(X)
            res[m] = X_imp

    # Compare the methods against the intact original data
    errors = {}
    for key, imputed_vals in res.items():
        # raw bias
        raw_bias = RB(pre_imputed_vals, imputed_vals)

        # percent bias
        perc_bias = PB(pre_imputed_vals, imputed_vals)

        # MAE
        mae = mean_absolute_error(pre_imputed_vals, imputed_vals)

        # RMSE
        rmse = root_mean_squared_error(pre_imputed_vals, imputed_vals)

        # R2 score
        r2 = r2_score(pre_imputed_vals, imputed_vals)

        errors[key] = [raw_bias, perc_bias, mae, rmse, r2]

        if print_on:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Imputation Method: {key}")
            print(f"RMSE: {rmse}")
            print(f"MAE: {mae}")
            print(f"Raw Bias: {raw_bias}")
            print(f"Percent Bias: {perc_bias}")
            print(f"R2: {r2}")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++")

    return errors
