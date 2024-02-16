import numpy as np
import matplotlib.pylab as plt

import xgboost as xgb

from sklearn.metrics import mean_squared_error as MSE

from ray import tune
from ray.air import session
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch

from pytorch_timeseries_loader.timeseries_loader import split_timeseries_data


def objective(config, data):
    X_train, y_train, X_test, y_test = split_timeseries_data(data,
                                                             sequence_len=24)

    train_set = xgb.DMatrix(X_train.squeeze(2),
                            label=y_train.squeeze(2))
    test_set = xgb.DMatrix(X_test.squeeze(2),
                           label=y_test.squeeze(2))

    res = {}
    _ = xgb.train(config,
                  train_set,
                  evals=[(test_set, "eval")],
                  evals_result=res,
                  verbose_eval=False
                  )
    # prediction = bst.predict(test_set)
    # rmse_ = np.sqrt(MSE(y_test.squeeze(2), prediction))
    rmse = res["eval"]["rmse"][-1]
    # session.report({"rmse": rmse, "done": True})
    session.report({"rmse": rmse})


def optimizeXGBoost(X):
    search_space = {"eval_metric": ["rmse"],
                    "objective": "reg:squarederror",
                    "max_depth": tune.randint(1, 9),
                    "min_child_weight": tune.choice([1, 2, 3]),
                    "subsample": tune.uniform(0.5, 1.0),
                    "eta": tune.loguniform(1e-4, 1e-1)
                    }

    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=5)
    scheduler = AsyncHyperBandScheduler()

    tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(objective,
                                         data=X),
                    resources={"cpu": 10,
                               "gpu": 1}),
                tune_config=tune.TuneConfig(metric="rmse",
                                            mode="min",
                                            search_alg=algo,
                                            scheduler=scheduler,
                                            num_samples=10,
                                            ),
                param_space=search_space,
                )
    results = tuner.fit()

    print(results.get_best_result(metric="rmse", mode="min").config)
    _ = results.get_best_result(metric="rmse",
                                mode="min").config.pop('eval_metric')
    return results.get_best_result(metric="rmse", mode="min").config


def XGBoostPredict(X, params):
    X_train, y_train, X_test, y_test = split_timeseries_data(X,
                                                             sequence_len=24)

    train_set = xgb.DMatrix(X_train.squeeze(2),
                            label=y_train.squeeze(2))
    test_set = xgb.DMatrix(X_test.squeeze(2),
                           label=y_test.squeeze(2))

    res = {}
    bst = xgb.train(params,
                    train_set,
                    evals=[(test_set, "eval")],
                    evals_result=res,
                    verbose_eval=False
                    )

    prediction = bst.predict(test_set)
    rmse = np.sqrt(MSE(y_test.squeeze(2), prediction))
    
    for i in range()
    plt.plot(y_test.squeeze(2), label="prediction")
    plt.plot(prediction, y, label="original")
    return prediction, rmse


if __name__ == "__main__":
    t = np.linspace(-np.pi, np.pi, 1000)
    y = np.sin(2.*np.pi*t*5)    # + np.random.normal(0, 1, 1000)
    params = optimizeXGBoost(y)

    yhat, error = XGBoostPredict(y, params)

    plt.show()
