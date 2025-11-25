# from sklearn.svm import LinearSVC
# from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel
# from sklearn.ensemble import RandomForestClassifier

from typing import Union
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import classification_report

class BaseEnsembleClassifier:
    def __init__(self, base_estimators : list, final_estimator, pickle_file_path : str):
        self.base_estimators = base_estimators
        self.base_estimator_info = {}
        self.final_estimator = final_estimator
        self.classification_report = {}
        self.final_estimator_dataset = pd.DataFrame([])
        self.final_est_params = pd.DataFrame({})
        self.pickle_file_path = pickle_file_path
        
        # Automatically load pickle file if it exists
        self.load_pickle_file()
        
    def load_pickle_file(self) -> bool:
        """
        Load weights/parameters from a pickle file and update the associated base estimators.
        
        The pickle file should contain a dictionary mapping estimator indices (or names)
        to their weights/parameters. This method will update the corresponding base
        estimators with the loaded weights.
        
        Returns:
            bool: True if file was loaded successfully, False otherwise.
        """
        if os.path.exists(self.pickle_file_path):
            try:
                with open(self.pickle_file_path, 'rb') as f:
                    weights_dict = pickle.load(f)
                
                # Update weights of associated models by indexing the pickle dictionary
                for idx, estimator in enumerate(self.base_estimators):
                    # Try to update using index first
                    if idx in weights_dict:
                        self._update_estimator_weights(estimator, weights_dict[idx])
                    # Also try using estimator name/type as key
                    elif hasattr(estimator, '__class__'):
                        estimator_name = estimator.__class__.__name__
                        if estimator_name in weights_dict:
                            self._update_estimator_weights(estimator, weights_dict[estimator_name])
                        # Try using string representation of index
                        elif str(idx) in weights_dict:
                            self._update_estimator_weights(estimator, weights_dict[str(idx)])
                
                # Store the loaded weights in base_estimator_info
                self.base_estimator_info = weights_dict
                print(f"Successfully loaded weights from {self.pickle_file_path}")
                return True
                
            except Exception as e:
                print(f"Error loading pickle file {self.pickle_file_path}: {e}")
                return False
        else:
            print(f"Pickle file not found: {self.pickle_file_path}")
            return False
    
    def _update_estimator_weights(self, estimator, weights):
        """
        Update the weights/parameters of an estimator.
        
        Parameters:
            estimator: The estimator object to update
            weights: Dictionary or object containing weights/parameters to set
        """
        try:
            # If weights is a dictionary, try to set parameters
            if isinstance(weights, dict):
                if hasattr(estimator, 'set_params'):
                    estimator.set_params(**weights)
                # If estimator has coef_ attribute, try to set it
                elif hasattr(estimator, 'coef_'):
                    if 'coef_' in weights:
                        estimator.coef_ = weights['coef_']
                # If estimator has feature_importances_, try to set it
                elif hasattr(estimator, 'feature_importances_'):
                    if 'feature_importances_' in weights:
                        estimator.feature_importances_ = weights['feature_importances_']
            # If weights is a numpy array, try to set as coef_ or feature_importances_
            elif isinstance(weights, np.ndarray):
                if hasattr(estimator, 'coef_'):
                    estimator.coef_ = weights
                elif hasattr(estimator, 'feature_importances_'):
                    estimator.feature_importances_ = weights
        except Exception as e:
            print(f"Warning: Could not update weights for {type(estimator).__name__}: {e}")

    
    def generate_classification_report(self, y_test, y_pred):
        """
        Generates, prints, and stores the classification report.
        
        Returns:
            pd.DataFrame: A dataframe version of the report for nice display.
        """
        # 1. Generate the dictionary format to store in the class attribute
        # output_dict=True allows us to save it as a usable object rather than just a string
        self.classification_report = classification_report(y_test, y_pred, output_dict=True)
        
        # 2. Print the standard text version for immediate feedback
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred))
        
        # 3. Return a DataFrame version (easier to read in Notebooks)
        # .transpose() makes the metrics (precision, recall) the columns and classes the rows
        return pd.DataFrame(self.classification_report).transpose()


    def _construct_meta_estimator_dataset(self, X):
        """
        Construct meta-features dataset from base estimator predictions.
        
        Parameters:
            X : array-like of shape (n_samples, n_features)
                Input samples.
        
        Returns:
            meta_features : array-like of shape (n_samples, n_base_estimators)
                Predictions from each base estimator.
        """
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Collect predictions from all base estimators
        meta_features_list = []
        for estimator in self.base_estimators:
            # Try predict_proba first (for probability outputs), else use predict
            if hasattr(estimator, 'predict_proba'):
                try:
                    proba = estimator.predict_proba(X)
                    # For binary classification, use probability of positive class
                    if proba.shape[1] == 2:
                        meta_features_list.append(proba[:, 1])
                    else:
                        # For multi-class, flatten probabilities
                        meta_features_list.append(proba.flatten())
                except:
                    # Fall back to predict if predict_proba fails
                    meta_features_list.append(estimator.predict(X))
            else:
                # Use class predictions
                meta_features_list.append(estimator.predict(X))
        
        # Stack predictions horizontally to create meta-features
        meta_features = np.column_stack(meta_features_list)
        return meta_features
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the ensemble model following sklearn conventions.
        
        Parameters:
            X : array-like of shape (n_samples, n_features)
                Training vectors.
            y : array-like of shape (n_samples,)
                Target values.
            sample_weight : array-like of shape (n_samples,), optional
                Sample weights. If None, the sample weights are initialized to 1.
        
        Returns:
            self : object
                Returns the instance itself.
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        
        # Step 1: Fit all base estimators
        for estimator in self.base_estimators:
            if sample_weight is not None and hasattr(estimator, 'fit'):
                try:
                    estimator.fit(X, y, sample_weight=sample_weight)
                except TypeError:
                    # Some estimators don't support sample_weight
                    estimator.fit(X, y)
            else:
                estimator.fit(X, y)
        
        # Step 2: Construct meta-features dataset from base estimator predictions
        self.final_estimator_dataset = self._construct_meta_estimator_dataset(X)
        
        # Step 3: Fit the final estimator on meta-features
        if sample_weight is not None and hasattr(self.final_estimator, 'fit'):
            try:
                self.final_estimator.fit(self.final_estimator_dataset, y, sample_weight=sample_weight)
            except TypeError:
                # Final estimator doesn't support sample_weight
                self.final_estimator.fit(self.final_estimator_dataset, y)
        else:
            self.final_estimator.fit(self.final_estimator_dataset, y)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
            X : array-like of shape (n_samples, n_features)
                Input samples.
        
        Returns:
            y_pred : array-like of shape (n_samples,)
                Predicted class labels.
        """
        # Step 1: Get predictions from base estimators
        meta_features = self._construct_meta_estimator_dataset(X)
        
        # Step 2: Use final estimator to make final predictions
        y_pred = self.final_estimator.predict(meta_features)
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
            X : array-like of shape (n_samples, n_features)
                Input samples.
        
        Returns:
            probabilities : array-like of shape (n_samples, n_classes)
                Class probabilities of the input samples.
        """
        if not hasattr(self.final_estimator, 'predict_proba'):
            raise AttributeError(
                f"Final estimator {type(self.final_estimator).__name__} does not support predict_proba"
            )
        
        # Step 1: Get predictions from base estimators
        meta_features = self._construct_meta_estimator_dataset(X)
        
        # Step 2: Use final estimator to get probabilities
        probabilities = self.final_estimator.predict_proba(meta_features)
        
        return probabilities
    
    def decision_function(self, X):
        """
        Compute decision function of samples in X.
        
        Parameters:
            X : array-like of shape (n_samples, n_features)
                Input samples.
        
        Returns:
            decision : array-like of shape (n_samples,) or (n_samples, n_classes)
                Decision function values.
        """
        if not hasattr(self.final_estimator, 'decision_function'):
            raise AttributeError(
                f"Final estimator {type(self.final_estimator).__name__} does not support decision_function"
            )
        
        # Step 1: Get predictions from base estimators
        meta_features = self._construct_meta_estimator_dataset(X)
        
        # Step 2: Use final estimator to get decision function
        decision = self.final_estimator.decision_function(meta_features)
        
        return decision
