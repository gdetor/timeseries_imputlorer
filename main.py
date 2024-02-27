# Main example of how to use Imputelorer for univariate time series data
# imputation. Copyright (C) 2024 Georgios Is. Detorakis
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

# import numpy as np
import matplotlib.pylab as plt

from sklearn.preprocessing import MinMaxScaler

from imputlorer.imputer import TSImputer
from imputlorer.generate_data import generateSinusoidalData
from imputlorer.xgb_regressor import optimizeXGBoost, XGBoostPredict


if __name__ == '__main__':
    # First step
    # Load or generate the data
    X, X0 = generateSinusoidalData(size=1000, missing_perc=0.2)

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

    exit()

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
        exit()
        prediction[key] = yhat
        error[key] = err
    plt.show()
