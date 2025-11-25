"""
Wrapper for sklearn's SVM with SHAP-based feature ranking via coef_ attribute.

This class wraps sklearn.svm.SVC and adds a coef_ attribute populated with
SHAP values for feature ranking. All SVM functionality is delegated to sklearn.
"""

import numpy as np
from sklearn.svm import SVC
import shap


class SVMSHAP(SVC):
    """
    Wrapper for sklearn's SVC with SHAP-based feature ranking.
    
    This is a thin wrapper around sklearn.svm.SVC that adds a coef_ attribute
    populated with SHAP values after fitting. All SVM functionality (predict,
    predict_proba, decision_function, etc.) comes directly from sklearn.
    
    The coef_ attribute is NOT set during initialization and is only available
    after calling fit(). It contains SHAP-based feature importance values for
    feature ranking purposes.
    
    Parameters:
        All parameters are passed directly to sklearn.svm.SVC.
        See sklearn documentation for parameter details.
    
    Attributes:
        coef_ : array-like
            SHAP-based feature importance values, computed after fit().
            Shape depends on number of classes. Only available after fit().
    
    Examples:
        >>> from custom_models.svm_shap import SVMSHAP
        >>> clf = SVMSHAP(kernel='rbf', C=1.0, probability=True)
        >>> clf.fit(X_train, y_train)
        >>> # coef_ now contains SHAP values for feature ranking
        >>> feature_importance = clf.coef_
        >>> y_pred = clf.predict(X_test)  # All sklearn methods work
    """
    
    def __init__(self, kernel='rbf', shap_features_limit=20,  **kwargs):
        """
        Initialize SVM wrapper. All kwargs are passed to sklearn.svm.SVC.
        
        Note: coef_ is NOT set here and will only be available after fit().
        """
        super().__init__(**kwargs)
        self.kernel = kernel
        self.shap_features_limit = shap_features_limit
        # coef_ will be populated after fit() with SHAP values
        self._shap_explainer = None
        self._coef_shap = None  # Store SHAP-based coefficients
    
    @property
    def coef_(self):
        """
        SHAP-based feature importance coefficients.
        
        Returns the SHAP-based coefficients if available, otherwise falls back
        to the parent class's coef_ (if it exists, e.g., for linear kernels).
        """
        if self._coef_shap is not None:
            return self._coef_shap
        # Fall back to parent's coef_ if it exists (for linear kernels)
        if hasattr(super(), 'coef_'):
            try:
                return super().coef_
            except AttributeError:
                pass
        raise AttributeError("coef_ has not been set. Call fit() first.")
    
    @coef_.setter
    def coef_(self, value):
        """Setter for coef_ to store SHAP-based values."""
        self._coef_shap = value
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the SVM model using sklearn and compute SHAP values for coef_.
        
        Parameters:
            X : array-like of shape (n_samples, n_features)
                Training vectors.
            y : array-like of shape (n_samples,)
                Target values.
            sample_weight : array-like of shape (n_samples,), optional
                Per-sample weights.
        
        Returns:
            self : object
                Returns the instance itself.
        """
        # Fit using sklearn's SVC
        super().fit(X, y, sample_weight=sample_weight)
        
        # Compute SHAP values and populate coef_ for feature ranking
        self._compute_shap_coef(X)
        
        return self
    
    def _compute_shap_coef(self, X):
        """
        Compute SHAP values using a very small, random sample of the training data.
        """
        X_np = np.array(X)
        n_total = len(X_np)

        # 1. Sample the data for explanation (for speed)
        n_explain = min(self.shap_features_limit, n_total)
        rng = np.random.default_rng(42) # Fixed seed for background sampling
        explain_indices = rng.choice(n_total, n_explain, replace=False)
        X_explain = X_np[explain_indices]
            
        # 2. Use a representative background sample (max 100 samples)
        n_background = min(100, n_total)
        background_indices = rng.choice(n_total, n_background, replace=False)
        background = X_np[background_indices]
        
        # print(f"SHAP explanation: {n_explain} samples. Background: {n_background} samples.")
        
        try:
            # Determine the model function based on probability setting
            model_func = self.predict_proba if self.probability else self.decision_function
                 
            # Initialize explainer
            self._shap_explainer = shap.KernelExplainer(model_func, background)
            
            # Compute SHAP values only for the sampled data
            shap_values = self._shap_explainer.shap_values(X_explain)
            
            # --- Calculate Feature Importance ---
            if isinstance(shap_values, list):
                # Multi-class: average across classes and samples
                shap_values_array = np.array(shap_values)
                feature_importance = np.mean(np.abs(shap_values_array), axis=(0, 1))
            else:
                # Binary: average absolute SHAP values across samples
                feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Store result: must be (1, n_features) shape for RFE/SFS compatibility
            self.coef_ = feature_importance.reshape(1, -1)
                
        except Exception as e:
            # Fallback: set coef_ to zeros if SHAP fails
            print(f"Warning: SHAP computation failed. Setting coef_ to zeros. Error: {e}")
            n_features = X_np.shape[1]
            # Must handle both multi-class and binary output shapes if possible
            n_classes = len(np.unique(self.classes_))
            output_shape = (n_classes, n_features) if n_classes > 2 else (1, n_features)
            self.coef_ = np.zeros(output_shape)
        except Exception as e:
            # Fallback: set coef_ to zeros if SHAP fails
            print(f"Warning: SHAP computation failed: {e}. Setting coef_ to zeros.")
            n_features = X.shape[1]
            self.coef_ = np.zeros((1, n_features))


   
