import os
import sys
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.pycache_prefix = os.path.join(root_dir, 'pycache', 'demo')
lore = __import__('imputlorer')


if __name__ == '__main__':
    # First step
    # Load or generate the data
    X, X0 = lore.generate_data.generateSinusoidalData(size=1000,
                                                      missing_perc=0.2)

    # Second step
    # Run all the imputation methods and collect the data in a dictionary
    standalone_method = ["nocb", "locf", "knn"]
    method = ["simple", "multi"]
    strategy = ["mean", "median", "constant"]

    res = {}
    for i, m in enumerate(method):
        for j, s in enumerate(strategy):
            imp = lore.imputer.TSImputer(method=m, strategy=s)
            X_imp = imp.run(X)
            res[m+"-"+s] = X_imp

    for i, m in enumerate(standalone_method):
        imp = lore.imputer.TSImputer(method=m)
        X_imp = imp.run(X)
        res[m] = X_imp

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
        params = lore.xgb_regressor.optimizeXGBoost(x,
                                                    sequence_len=sequence_len)
        yhat, err = lore.xgb_regressor.XGBoostPredict(x,
                                                      params,
                                                      sequence_len=sequence_len)
        prediction[key] = yhat
        error[key] = err

    print("************************************************************")
    for method, err in error.items():
        print(f"Imputation method: {method} had an RMSE: {err}")
    print("************************************************************")
    plt.show()
