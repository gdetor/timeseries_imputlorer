import pandas as pd
import matplotlib.pylab as plt

from imputlorer.imputer import TSImputer
from imputlorer.load import generateSyntheticData


if __name__ == '__main__':
    X = generateSyntheticData(size=100, missing_perc=0.2)
    X_copy = X.copy()

    imp = TSImputer(method="nocb")
    a, b = imp.getDataRange(X)
    print(f"Range [{a}, {b}]")
    print(f"Med-range = {imp.getDataMidRange(X)}")
    X_imp = imp.run(X)

    dfX = pd.DataFrame(X)
    nul_data = pd.isnull(dfX)
    dfX = dfX.assign(FillMean=dfX.fillna(dfX.mean()))

    imp0 = TSImputer(method="simple", strategy="median")
    X_imp0 = imp0.run(X.T.copy())

    imp1 = TSImputer(method="multi", strategy="median")
    X_imp1 = imp1.run(X.T.copy())

    imp2 = TSImputer(method="knn")
    X_imp2 = imp2.run(X.T.copy())

    plt.plot(X_copy, '-', label='original')
    plt.plot(X_imp0, 'x--', label='simple', ms=10)
    plt.plot(X_imp1, 'o--', label='multi')
    plt.plot(X_imp2, 'p--', label='knn')
    plt.plot(X, label="damaged")
    plt.legend()
    plt.show()
