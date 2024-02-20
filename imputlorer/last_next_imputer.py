import numpy as np


class LastNextValueImputer:
    """ Next observation carried backward. This class is similar to Pandas
    bfill() method.
        Last observation carried forwrard. This class is similar to Pandas
    ffill() method.
    """
    def __init__(self,
                 strategy="locf",   # or nocb
                 missing_values=np.nan,
                 copy=True):
        self.strategy = strategy
        self.missing_values = missing_values
        self.copy = copy

    def fit_transform(self, X):
        if self.copy:
            x = X.copy()
        else:
            x = X

        if self.strategy == "nocb":
            x = np.flip(x)
        # FIXME Make the following statement more generic (include None, 0,
        # etc)
        mask_ = np.isnan(x)
        index_ = np.where(~mask_,
                          np.arange(mask_.shape[1]), 0)
        np.maximum.accumulate(index_, axis=1, out=index_)
        x[mask_] = x[np.nonzero(mask_)[0], index_[mask_]]

        if self.strategy == "nocb":
            x = np.flip(x)
        x = x.T
        if self.strategy == "nocb" and x[0] == np.nan:
            x[0] = x[1]
        if self.strategy == "nocb" and x[-1] == np.nan:
            x[-1] = x[0]
        if self.strategy == "locf" and x[-1] == np.nan:
            x[-1] = x[-2]
        if self.strategy == "locf" and x[0] == np.nan:
            x[0] = x[-1]
        return x
