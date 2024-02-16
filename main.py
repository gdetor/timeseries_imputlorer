# import pandas as pd
import matplotlib.pylab as plt

from imputlorer.imputer import TSImputer
from imputlorer.generate_data import generateSyntheticData


if __name__ == '__main__':
    # First step
    # Load or generate the data
    X, X0 = generateSyntheticData(size=100, missing_perc=0.2)

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
            res[m+s] = X_imp

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
    plt.show()
