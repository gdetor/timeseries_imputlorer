import numpy as np
# import pandas as pd

from sklearn.linear_model import BayesianRidge, Ridge, LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer

from .last_next_imputer import LastNextValueImputer


class TSImputer:
    def __init__(self,
                 method="simple",
                 missing_values=np.nan,
                 strategy="mean",
                 fill_value=0.0,
                 estimator=BayesianRidge,
                 sample_posterior=False,
                 max_iter=10,
                 tol=1e-3,
                 n_nearest_features=None,
                 imputation_order="ascending",
                 skip_complete=False,
                 min_value=-np.inf,
                 max_value=np.inf,
                 random_state=None,
                 n_neighbors=5,
                 weights="uniform",
                 metric="nan_euclidean",
                 ):
        self.missing_values = missing_values
        if method == "nocb":
            self.imputer = LastNextValueImputer(strategy=method,
                                                missing_values=missing_values,
                                                )
        elif method == "locf":
            self.imputer = LastNextValueImputer(strategy=method,
                                                missing_values=missing_values,
                                                )
        elif method == "simple":
            self.imputer = SimpleImputer(missing_values=missing_values,
                                         strategy=strategy,
                                         fill_value=fill_value
                                         )
        elif method == "multi":
            estimator = BayesianRidge()
            estimator = LinearRegression()
            estimator = Ridge()
            self.imputer = IterativeImputer(missing_values=missing_values,
                                            initial_strategy=strategy,
                                            fill_value=fill_value,
                                            estimator=estimator,
                                            sample_posterior=sample_posterior,
                                            max_iter=max_iter,
                                            tol=tol,
                                            n_nearest_features=n_nearest_features,
                                            imputation_order=imputation_order,
                                            skip_complete=skip_complete,
                                            min_value=min_value,
                                            max_value=max_value,
                                            random_state=random_state
                                            )
        elif method == "knn":
            self.imputer = KNNImputer(missing_values=missing_values,
                                      n_neighbors=n_neighbors,
                                      weights=weights,
                                      metric=metric
                                      )
        else:
            print("Not a valid method!")
            print("Choose simple, multi, or knn!")
            exit()

    def run(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return self.imputer.fit_transform(X)

    def reset(self):
        pass

    def getDataRange(self, x):
        return np.nanmin(x), np.nanmax(x)

    def getDataMidRange(self, x):
        x_min, x_max = self.getDataRange(x)
        return (x_max + x_min) / 2

    def getStatistics(self):
        return self.imputer.statistics_

    def getIndicator(self):
        return self.imputer.indicator_

    def getNumFeatures(self):
        return self.imputer.n_features_in_

    def getFeaturesNames(self):
        return self.imputer.feature_names_in_
