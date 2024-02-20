# import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.preprocessing import MinMaxScaler

from imputlorer.imputer import TSImputer
from imputlorer.generate_data import generateSyntheticData
from imputlorer.xgb_regressor import optimizeXGBoost, XGBoostPredict


if __name__ == '__main__':
    # First step
    # Load or generate the data
    X, X0 = generateSyntheticData(size=100, missing_perc=0.2)
    X[0] = np.nan

    # Second step
    # Run all the imputation methods and collect the data in a dictionary
    standalone_method = ["nocb", "locf", "knn"]
    method = ["simple", "multi"]
    strategy = ["mean", "median", "constant"]

    res = {}
    for i, m in enumerate(method):
        for j, s in enumerate(strategy):
            imp = TSImputer(method=m, strategy=s)
            X_imp = imp.run(X)
            res[m+"-"+s] = X_imp

    for i, m in enumerate(standalone_method):
        imp = TSImputer(method=m)
        X_imp = imp.run(X)
        res[m] = X_imp

    # dfX = pd.DataFrame(X)
    # nul_data = pd.isnull(dfX)
    # dfX = dfX.assign(FillMean=dfX.fillna(dfX.mean()))

    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)
    ax.plot(X0, '-', c='k', label='original')
    ax.plot(X, label="damaged", c='r')
    for key, item in res.items():
        ax.plot(item, label=key)
    ax.legend()

    # Third step
    # Optimize and train the neural networks on the imputed and original data
    # Collect all the errors for comparing
    print("Running XGBoost regression on imputing data!")
    sequence_len = 4
    prediction = {}
    error = {}
    for key, item in res.items():
        print(f"============ Now running {key} =============")
        x = item
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x.reshape(-1, 1))[:, 0]
        params = optimizeXGBoost(x, sequence_len=sequence_len)
        yhat, err = XGBoostPredict(x, params, sequence_len=sequence_len)
        prediction[key] = yhat
        error[key] = err

    plt.show()
