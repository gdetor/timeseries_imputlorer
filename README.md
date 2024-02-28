# timeseries_imputlorer

This repository contains scripts that implement univariate time series imputation
methods. The user has the ability of testing the methods and evaluating them
using statistical tools and standard metrics such as mean absolute error. 
In addition, the user can test any provided imputation method using XGBoost
regression. 

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
│   ├── pytorch_timeseries_loader
│   │   ├── example.py
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── timeseries_loader.py
│   │   └── transforms.py
│   └── xgb_regressor.py
├── LICENSE
├── main.py
├── README.md
└── test
    ├── test_comparisons.py
        └── test_imputer.py

```



## Example usage


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
