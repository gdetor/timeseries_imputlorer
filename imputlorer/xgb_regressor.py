# This script contains functions for the XGBoost regressor and its
# hyperparameters tuning using Ray.
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
import matplotlib.pylab as plt

import xgboost as xgb

from sklearn.metrics import mean_squared_error as MSE

from ray import tune
from ray.air import session
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch

from .pytorch_timeseries_loader.timeseries_loader import split_timeseries_data


def objective(config, data, sequence_len):
    """! It is the objective function for XGBoost regressor hyperparameters
    tuning. It estimates the root mean squared error on the test data, which
    the Ray Tune will try to minimize.

    @param config A Python dictionary that contains a Ray search space defined
    by the user.
    @param data The univariate time series data on which the XGBoost regressor
    will be trained.
    @param sequence_len The length of the window containing the past data
    points (historical data).

    @return void
    """
    # Split the training/test data set
    X_train, y_train, X_test, y_test = split_timeseries_data(
            data,
            sequence_len=sequence_len,
            horizon=1,
            univariate=True,
            torch=False)

    # Create the appropriate XGB matrices
    train_set = xgb.DMatrix(data=X_train, label=y_train)
    test_set = xgb.DMatrix(data=X_test, label=y_test)

    # The dictionary results holds the regression results
    results = {}
    # Call the XGB train method
    _ = xgb.train(config,
                  train_set,
                  evals=[(test_set, "eval")],
                  evals_result=results,
                  verbose_eval=False
                  )
    # Get the RMSE
    rmse = results["eval"]["rmse"][-1]

    # Report the RMSE to Ray Tune
    session.report({"rmse": rmse})


def optimizeXGBoost(X, sequence_len=8):
    """! It is the XGBoost regressor's primary optimization function. This
    function defines the XGBoost hyperparameter search space and calls all the
    necessary tasks for tuning the hyperparameter.

    @param X Input ndarray of shape (n_samples, ).
    @param sequence_len The length of the window containing the past data
    points (historical data).

    @return A Python dictionary that contains the optimal hyperparameters.
    """

    # Define the search space of hyperparameters
    search_space = {"eval_metric": ["rmse"],
                    "objective": "reg:squarederror",
                    "max_depth": tune.randint(1, 9),
                    "min_child_weight": tune.choice([1, 2, 3]),
                    "subsample": tune.uniform(0.5, 1.0),
                    "eta": tune.loguniform(1e-4, 1e-1),
                    }

    # Set the Optuna optimization algorithm, and the scheduler
    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=5)
    scheduler = AsyncHyperBandScheduler()

    # Instantiate the Tuner class
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
    # Run the optimization
    results = tuner.fit()

    # Gather the results
    print(results.get_best_result(metric="rmse", mode="min").config)
    _ = results.get_best_result(metric="rmse",
                                mode="min").config.pop('eval_metric')
    return results.get_best_result(metric="rmse", mode="min").config


def XGBoostPredict(X, params, sequence_len=8, display=False):
    """! This function trains an XGBoost regressor using the optimal
    hyperparameters, and it returns the prediction values and the
    corresponding RMSE.

    @param X A ndarray of shape (n_samples, ).
    @param params A Python dictionary that contains all the XGBoost
    hyperparameters (dict).
    @param sequence_len The length of the window containing the past data
    points (historical data).
    @param display If True, it plots the test and the predicted data.

    @return A tuple that contains the predictions and the corresponding rmse
    (ndarray, float).
    """
    # Split training/test data set
    X_train, y_train, X_test, y_test = split_timeseries_data(
            X,
            sequence_len=sequence_len,
            horizon=1,
            univariate=True,
            torch=False)

    # Create XGB matrices
    train_set = xgb.DMatrix(data=X_train, label=y_train)
    test_set = xgb.DMatrix(data=X_test, label=y_test)

    # Train the XBGoost regressor using the optimal hyperparameters
    res = {}
    bst = xgb.train(params,
                    train_set,
                    evals=[(test_set, "eval")],
                    evals_result=res,
                    verbose_eval=False
                    )

    # Estimate a prediction
    prediction = bst.predict(test_set)
    rmse = np.sqrt(MSE(y_test, prediction))

    # Plot the results
    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_test[:, 0], label="prediction")
        ax.plot(prediction, label="original")
        ax.legend()
    return prediction, rmse
