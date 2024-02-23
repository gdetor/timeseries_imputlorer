import numpy as np
from scipy import stats
# import matplotlib.pylab as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

from imputlorer.imputer import TSImputer


def summaryStatistics(X):
    X_ = X[~np.isnan(X)]
    mu = np.mean(X_)
    sd = np.std(X_)
    med = np.median(X_)
    mod = stats.mode(X_)
    # qut = stat.quantiles(X)
    per = np.percentile(X_, [25, 50, 75])

    missing_vals = np.isnan(X).sum()

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

    kyrtosis = stats.kurtosis(X_, axis=0, bias=True)
    skewness = stats.skew(X_, axis=0, bias=True)
    print(f"Kurtosis: {kyrtosis} and skewness: {skewness}")

    CI = stats.t.interval(confidence=0.9,
                          df=len(X_)-1,
                          loc=mu,
                          scale=stats.sem(X_))
    print(f"Confidence Interval: {CI}")
    print("===============================")


def RB(y, yhat):
    """ Raw bias """
    return np.mean(y - yhat)


def PB(y, yhat):
    """ Percent bias"""
    return 100.0 * np.mean((y - yhat) / np.abs(y))


def compareImputationMethods(X,
                             method=["simple"],
                             standalone_method=["nocb"],
                             strategy=["mean"],
                             missing_vals_perc=0.2,
                             print_on=True):

    n_missing_vals = int(missing_vals_perc * len(X))
    missing_vals_mask = np.random.choice([i for i in range(len(X))],
                                         size=n_missing_vals)
    pre_imputed_vals = X[missing_vals_mask]
    X[missing_vals_mask] = np.nan

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

    for key, item in res.items():
        y = item[:, 0]
        imputed_vals = y[missing_vals_mask]

        is_const_arr = np.all(imputed_vals == imputed_vals[0])

        # t-test
        ttest_res = stats.ttest_rel(pre_imputed_vals,
                                    imputed_vals,
                                    axis=0,
                                    alternative='two-sided')

        # Pearson correlation
        if is_const_arr is False:
            rho = stats.pearsonr(pre_imputed_vals, imputed_vals)

        # raw bias
        raw_bias = RB(pre_imputed_vals, imputed_vals)

        # percent bias
        perc_bias = PB(pre_imputed_vals, imputed_vals)

        # MAE
        mae = mean_absolute_error(pre_imputed_vals, imputed_vals)

        # RMSE
        rmse = np.sqrt(mean_squared_error(pre_imputed_vals, imputed_vals))
        if print_on:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Imputation Method: {key}")
            print(f"RMSE: {rmse}")
            print(f"MAE: {mae}")
            print(f"Raw Bias: {raw_bias}")
            print(f"Percent Bias: {perc_bias}")
            print(f"t: {ttest_res.statistic}, p-value: {ttest_res.pvalue}")
            if is_const_arr is False:
                print(f"Pearson rho: {rho.statistic}, p-value: {rho.pvalue}")

            print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
