# Timeseries Imputlorer

This repository contains scripts that implement univariate time series
imputation methods. The user can test the methods and evaluate them using
statistical tools and standard metrics such as mean absolute error. In addition,
the user can try any provided imputation method using XGBoost regression.

> The main goal of the *timeseries_imputlorer* is to provide tools and means to evaluate different imputation methods on any univariate time series data set.

The main class named `TSImputer` compiles five basic imputation methods:
  1. Next Observation Carried Backward (NOCB), which is similar to `bfill()`
  of Pandas,
  2. Last Observation Carried Forward (LOCF), which is similar to `ffill()` of
  Pandas,
  3. SimpleImputer from Sklearn,
  4. IterativeImputer of Sklearn,
  5. KNNImputer of Sklearn.

> Some of the scripts in this repository use the dataloader provided by the repository **[pytorch_timeseries_dataloader](https://github.com/gdetor/pytorch_timeseries_loader)**. The XGBoost trainer uses the dataloader to split the training/test data set.


## Contents

The repository is organized as follows:
```
imeseries_imputlorer/
├── data
│   ├── synthetic_damaged_normal.npy
│   └── synthetic_original_normal.npy
├── demo
│   ├── compare_methods.py
│   ├── compare_methods_xgb.py
│   ├── data_imputation.py
│   └── __init__.py
├── imputlorer
│   ├── compare_imputation_methods.py
│   ├── generate_data.py
│   ├── imputer.py
│   ├── __init__.py
│   ├── nocb_locf_imputer.py
│   ├── pytorch_timeseries_loader/
│   └── xgb_regressor.py
├── LICENSE
├── main.py
├── README.md
└── test
    ├── test_comparisons.py
    └── test_imputer.py

```
The core implementation lies in the directory **imputlorer**. There, the user can find the following files:
  - **imputer** This file contains the class `TSImputer`, which implements the imputation methods.
  - **nocb_locf_imputer.py** This is the implementation of the *NOCB* and *LOCF* methods.
  - **generate_data.py** In this script, the user will find two functions that generate white noise and sinusoidal synthetic data for testing purposes. The user can ignore these functions and use their own.
  - **compare_imputation_methods.py** This file contains a function that estimates and reports the summary statistics of the data (pre- and post-imputation), and a function that compares imputation methods and reports the error metrics.
  - **xgb_regressor.py** In this file, there are all the functions so that the user can optimize the hyperparameters of an XGBoost regressor and then test imputation methods using the quality (error) of the predictions of the regressor as a measure of evaluation.
  - The directory **pytorch_timeseries_loader** is the data loader to split the training/test data set for the XGBoost regressor. 

The directory **tests** includes tests for the class `TSImputer` and the evaluation tools. Finally, the folder **demo** contains three examples showing how to perform a data imputation, compare and evaluate imputation methods, and use XGBoost regression to evaluate different imputation methods.


## Install

To be able to use all the classes, methods, and functions provided in this repository, the user first has to clone the present repository
recursively so they can also get the **pytorch_timeseries_loader** submodule.

```bash
$ git clone https://github.com/gdetor/timeseries_imputlorer --recursive
```
Once, they have cloned the repository, they can install all the dependencies by
typing and running the following commands in a terminal

```bash
$ cd timeseries_imputlorer
$ pip (or pip3) install -Ur requirements.txt
```


## Example usage

Remember that the user can use their data in all the examples below. The only constraint is that the data must be stored in Numpy arrays of shape (n_samples, 1) (since we treat univariate time series and thus there is only one feature). Moreover, the following code snippets are available as standalone Python scripts under the folder `data/`.

### Time series imputation

The primary function of **Imputlorer** is to apply different imputation methods on univariate time series data and evaluate and compare them. Thus, the first example shows how to perform data imputation on synthetic data and, more precisely, sinusoidal data with normal noise. The code snippet below demonstrates the three main steps the user has to take to do the imputation and obtain visual results for further inspection. Finally, the code shows how to print out the summary statistics of the data after imputation to compare the effect of different 
imputation techniques have on the data.

```python
    # First step
    # Load or generate the data
    X, X0 = lore.generate_data.generateSinusoidalData(size=1000,
                                                      missing_perc=0.2)

    # Second step
    # Run all the imputation methods and collect the data in a dictionary
    standalone_method = ["nocb", "locf", "knn"]
    method = ["simple", "multi"]
    strategy = ["mean", "median", "constant"]

	# Store the name of the method and the results in the dictionary res
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

    # Plot all the results for visual inspection
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)
    ax.plot(X0, '-', c='k', label='original')
    ax.plot(X, label="damaged", c='r')
    for key, item in res.items():
        ax.plot(item, label=key)
    ax.legend()

    # Report summary statistics for imputed data
    for key, item in res.items():
        lore.compare_imputation_methods.summaryStatistics(item)
```

### Compare and evaluate imputation methods

The second example demonstrates how the user can compare several imputation methods. Again, the code snippet below shows the two steps the user must take to compare the methods. The function `compareImputationMethods` will return a  Python dictionary containing the name of each tested method and its corresponding errors.

```python
  	# First step
    # Load or generate the data
    X, X0 = lore.generate_data.generateSinusoidalData(size=100,
                                                      missing_perc=0.2)

    # Second step
    # Run all the imputation methods and collect the data in a dictionary
    standalone_method = ["nocb", "locf", "knn"]
    method = ["simple", "multi"]
    strategy = ["mean", "median", "constant"]

    err = lore.compare_imputation_methods.compareImputationMethods(
            X0,
            method=method,
            standalone_method=standalone_method,
            strategy=strategy,
            missing_vals_perc=0.2,
            print_on=True)

    # The dictionary err contains the errors: Raw bias, percent bias, MAE,
    # RMSE, and R2
    # print(err)
```


### Use XGBoost regression to evaluate different imputation methods

The third and last example shows how to use XGBoost regression to evaluate the performance of different imputation methods, and thus, the user can decide which is suitable for their data.
In the current code snippet, all the available imputation techniques are tested. First, the raw data (with NaN values) are imputed, and then the same data is used to train XGBoost regressors. The XGBoost regressor's hyperparameters are tuned for each method using Ray Tune. Then, the optimal hyperparameters pass to the XGBoost trainer, which, after convergence, will provide all the test predictions and the corresponding errors. Hence, the user can decide which method achieved the best results in the regression and thus keep that method.

```python
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
        # Normalize the data
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x.reshape(-1, 1))[:, 0]

        # Tune the XGBoost hyperparameters using Ray
        params = lore.xgb_regressor.optimizeXGBoost(x,
                                                    sequence_len=sequence_len)

        # Make some test predictions and collect the error
        yhat, err = lore.xgb_regressor.XGBoostPredict(
                x,
                params,
                sequence_len=sequence_len)
        # Store the predictions and the errors for later use
        prediction[key] = yhat
        error[key] = err
```


## Dependencies
  - Sklearn 1.4.1
  - Scipy 1.12.0
  - Numpy 1.26.4
  - Matplotlib 3.5.1
  - XGBoost 1.7.4
  - Ray 2.3.1
  - XGBoost Ray 0.1.15

## Tested platforms

The software available in this repository has been tested on the following platforms:
  - Ununtu 22.04.4 LTS
      - Python 3.10.12
      - GCC 11.4.0
      - x86_64
