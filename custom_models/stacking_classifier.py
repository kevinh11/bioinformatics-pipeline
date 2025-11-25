from typing_extensions import override
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

from typing import Union
import numpy as np
import pandas as pd
from base.base_ensemble import BaseEnsembleClassifier


class StackingClassifier(BaseEnsembleClassifier):
    def __init__(self, base_estimators, final_estimator):
        super().__init__(base_estimators, final_estimator)
        
    def fit_base_estimators(self, x_train : Union[pd.DataFrame, np.ndarray] , y_train : Union[pd.DataFrame, np.ndarray]) -> None:
        # Fit all base models
        for estimator in self.base_estimators:
            estimator.fit(x_train, y_train)

    def fit_final_estimator(self, x_train : Union[pd.DataFrame, np.ndarray], y_train : Union[pd.DataFrame, np.ndarray] ):

        self.final_estimator.fit(x_train, y_train)
        return self
    
    
    def predict(self, x_test : Union[pd.DataFrame, np.ndarray], y_test : Union[pd.DataFrame, np.ndarray] ):
        
        for est in self.estimators:
            try:
                y_pred_proba = est.predict_proba(x_test)
                result_df = pd.DataFrame.from_dict({
                    "probability" : y_pred_proba, 
                    "model_name" : est.__name__,
                    "endpoint" : 1 if y_pred_proba >= 0.5 else 0
                })
                self.final_est_params = pd.concat([self.final_est_params, result_df])
                self.final_estimator.fit(x_train, y_train)
            except Exception as e:
                print(e)


        
