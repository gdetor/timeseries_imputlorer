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
        self.strategy_cp = strategy
        self.missing_values = missing_values
        self.copy = copy
        self.correct_nans = False

    def fit_transform(self, X):
        x = self.transform(X)
        if np.isnan(x[0, :]).sum() > 0:
            if np.isnan(x[0, 0]) and self.strategy == "locf":
                self.strategy = "nocb"
                x = self.transform(x)
            if np.isnan(x[0, -1]) and self.strategy == "locf":
                self.stategy = "locf"
                x = self.transform(x)
            if np.isnan(x[0, 0]) and self.strategy == "nocb":
                self.strategy = "nocb"
                x = self.transform(x)
            if np.isnan(x[0, -1]) and self.strategy == "nocb":
                self.correct_nans = True
                self.stategy = "locf"
                x = self.transform(x)
            self.reset()
        x = x.T
        return x

    def reset(self):
        self.strategy = self.strategy_cp
        self.correct_nans = False

    def transform(self, X):
        if self.copy:
            x = X.copy()
        else:
            x = X

        if self.strategy == "nocb" and self.correct_nans is False:
            x = np.flip(x)
        # FIXME Make the following statement more generic (include None, 0,
        # etc)
        mask_ = np.isnan(x)
        index_ = np.where(~mask_,
                          np.arange(mask_.shape[1]), 0)
        np.maximum.accumulate(index_, axis=1, out=index_)
        x[mask_] = x[np.nonzero(mask_)[0], index_[mask_]]

        if self.strategy == "nocb" and self.correct_nans is False:
            x = np.flip(x)
        return x


if __name__ == "__main__":
    x = np.array([[np.nan, np.nan, 2.3, 1.5, 5.5, np.nan]])
    print(x)

    ln = LastNextValueImputer(strategy="locf")
    y = ln.fit_transform(x)

    print(y)
