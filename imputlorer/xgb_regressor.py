import numpy as np
import matplotlib.pylab as plt

import xgboost as xgb

from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import MinMaxScaler

from ray import tune
from ray.air import session
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch

from .pytorch_timeseries_loader.timeseries_loader import split_timeseries_data


def objective(config, data, sequence_len):
    X_train, y_train, X_test, y_test = split_timeseries_data(
            data,
            sequence_len=sequence_len,
            horizon=1,
            univariate=True,
            torch=False)

    train_set = xgb.DMatrix(data=X_train, label=y_train)
    test_set = xgb.DMatrix(data=X_test, label=y_test)

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


def optimizeXGBoost(X, sequence_len=8):
    search_space = {"eval_metric": ["rmse"],
                    "objective": "reg:squarederror",
                    "max_depth": tune.randint(1, 9),
                    "min_child_weight": tune.choice([1, 2, 3]),
                    "subsample": tune.uniform(0.5, 1.0),
                    "eta": tune.loguniform(1e-4, 1e-1),
                    }

    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=5)
    scheduler = AsyncHyperBandScheduler()

    tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(objective,
                                         data=X,
                                         sequence_len=sequence_len),
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


def XGBoostPredict(X, params, sequence_len=8, display=False):
    X_train, y_train, X_test, y_test = split_timeseries_data(
            X,
            sequence_len=sequence_len,
            horizon=1,
            univariate=True,
            torch=False)

    train_set = xgb.DMatrix(data=X_train, label=y_train)
    test_set = xgb.DMatrix(data=X_test, label=y_test)

    res = {}
    bst = xgb.train(params,
                    train_set,
                    evals=[(test_set, "eval")],
                    evals_result=res,
                    verbose_eval=False
                    )

    prediction = bst.predict(test_set)
    rmse = np.sqrt(MSE(y_test, prediction))
    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_test[:, 0], label="prediction")
        ax.plot(prediction, label="original")
        ax.legend()
    return prediction, rmse
