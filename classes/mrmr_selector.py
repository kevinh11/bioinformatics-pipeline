
import pandas as pd
from mrmr import mrmr_classif
from sklearn.base import BaseEstimator, TransformerMixin

class MRMRSelector(BaseEstimator, TransformerMixin):
    def __init__(self, K=50, show_progress=False):
        self.K = K
        self.show_progress = show_progress
        self.selected_features_ = None

    def fit(self, X, y):
        # mrmr_classif automatically uses mutual information and redundancy internally
        self.selected_features_ = mrmr_classif(
            X, y,
            K=self.K,
            show_progress=self.show_progress
        )
        return self

    def transform(self, X):
        # Ensure that the data type supports column selection
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            raise TypeError("MRMRSelector expects a pandas DataFrame as input.")
