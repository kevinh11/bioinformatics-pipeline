
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class CleanFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, missing_thresh=0.3, variance_thresh=1e-6):
        self.missing_thresh = missing_thresh
        self.variance_thresh = variance_thresh
        self.keep_features_ = []

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=X.columns if hasattr(X, "columns") else None)

        # Drop by missing %
        keep_missing = X.isna().mean() < self.missing_thresh
        X2 = X.loc[:, keep_missing]

        # Drop by variance
        var = X2.var()
        keep_var = var > self.variance_thresh

        self.keep_features_ = X2.columns[keep_var].tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=X.columns if hasattr(X, "columns") else None)
        return X[self.keep_features_]
