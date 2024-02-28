# timeseries_imputlorer

This repository contains scripts that implement univariate time series imputation
methods. The user has the ability of testing the methods and evaluating them
using statistical tools and standard metrics such as mean absolute error. 
In addition, the user can test any provided imputation method using XGBoost
regression.

> The main goal of the *timeseries_imputlorer* is to provide tools and means to evaluate different imputation methods on any kind of univariate time series data set.

The main class named `TSImputer` compiles five major imputation methods:
  1. Next Observation Carried Backward (NOCB), which is similar to `bfill()`
  of Pandas,
  2. Last Observation Carried Forward (LOCF), which is similar to `ffill()` of
  Pandas,
  3. SimpleImputer from Sklearn,
  4. IterativeImputer of Sklearn,
  5. KNNImputer of Sklearn.

The dataloader that is being used in training and testing the XGBoost regressor
can be found [here](https://github.com/gdetor/pytorch_timeseries_loader).


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
The core implementation lies in the directory **imputlorer**. There the user can find the following files:
  - **imputer** This file contains the class `TSImputer`, which implements the imputation methods.
  - **nocb_locf_imputer.py** This is the implementation of the *NOCB* and *LOCF* methods.
  - **generate_data.py** In this script there are two functions that generate white noise and sinusoidal synthetic data for testing purposes. The user can ignore these functions and use their own.
  - **compare_imputation_methods.py** This file contains a function that estimates and reports the summary statistics of the data (pre- and post-imputation), and a function that compares imputation methods and reports the error metrics.
  - **xgb_regressor.py** In this file there are all the functions so that the user can optimize the hyperparameters of an XGBoost regressor and then test imputation methods using the quiality (error) of the predictions of the regressor as a measure of evaluation.
  - The directory **pytorch_timeseries_loader** is the dataloader used to split the training/test data set for the XGBoost regressor. 

In the directory **tests** there are tests for both the class `TSImputer` and the evaluation tools. Finally, the folder **demo** contains three examples showing how to perform a data imputation, compare and evaluate imputation methods, and how to use XGBoost regression to evaluate different imputation methods.

## Example usage

### Time series imputation


### Compare and evaluate imputation methods


### Use XGBoost regression to evaluate different imputation methods


## Dependencies
  - Sklearn 1.4.1
  - Scipy 1.12.0
  - Numpy 1.26.4
  - Matplotlib 3.5.1
  - XGBoost 1.7.4
  - Ray 2.3.1
  - XGBoost Ray 0.1.15

## Tested platforms

The software available in this repository has been tested in the following platforms:
  - Ununtu 22.04.4 LTS
      - Python 3.10.12
      - GCC 11.4.0
      - x86_64
