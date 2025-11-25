import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    make_scorer,
    precision_score, 
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    recall_score,
    confusion_matrix, 
    f1_score
)
from sklearn.feature_selection import RFE, SelectKBest, SequentialFeatureSelector, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import Union
from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu
from sklearn.feature_selection import mutual_info_classif, f_classif 
from utils.plotter import  plot_roc_curve, plot_pr_curve
import numpy as np
import os

# --- Feature Extraction Helpers (Kept from your original code) ---

def get_selected_features(estimator, feature_names):
    """Extract selected feature names from any fitted estimator or selector."""
    selected = None

    if hasattr(estimator, "named_steps"):  # Pipeline
        selector = estimator.named_steps.get('selector')
        if selector is not None:
            return get_selected_features(selector, feature_names)

    if hasattr(estimator, "get_support"):
        try:
            support = estimator.get_support()
            selected = np.array(feature_names)[support].tolist()
        except Exception:
            pass
    elif hasattr(estimator, "coef_"):
        try:
            coef = np.ravel(estimator.coef_)
            nonzero = np.abs(coef) > 1e-6
            selected = np.array(feature_names)[nonzero].tolist()
        except Exception:
            pass
    elif hasattr(estimator, "estimator_"):
        return get_selected_features(estimator.estimator_, feature_names)

    return selected or []


def get_filtered_features(filter_method: str, x_tr: pd.DataFrame, y_tr: pd.Series) -> list:
    """Return feature names using a univariate filter."""
    if filter_method is None:
        return list(x_tr.columns)
    
    respond_mask = y_tr == 1
    nonrespond_mask = y_tr == 0

    responders = x_tr.loc[respond_mask]
    nonresponders = x_tr.loc[nonrespond_mask]

    p_thresh = 0.1
    significant = []

    # --- 1. t-test ---
    if filter_method == "ttest":
        for col in x_tr.columns:
            try:
                _, p = ttest_ind(responders[col].dropna(),
                                 nonresponders[col].dropna(),
                                 equal_var=False)
                if p < p_thresh:
                    significant.append(col)
            except:
                continue

    # --- 2. KS test ---
    elif filter_method == "kolmogorov_smirnov":
        for col in x_tr.columns:
            try:
                _, p = ks_2samp(responders[col].dropna(),
                                nonresponders[col].dropna())
                if p < p_thresh:
                    significant.append(col)
            except:
                continue

    # --- 3. Mann-Whitney U (non-parametric t-test) ---
    elif filter_method == "mannwhitney":
        for col in x_tr.columns:
            try:
                _, p = mannwhitneyu(responders[col].dropna(),
                                    nonresponders[col].dropna(),
                                    alternative="two-sided")
                if p < p_thresh:
                    significant.append(col)
            except:
                continue

    # --- 4. Variance threshold ---
    elif filter_method == "variance":
        variances = x_tr.var()
        significant = variances[variances > 1e-4].index.tolist()

    # --- 5. Mutual information (top_k best features) ---
    elif filter_method == "mutual_info":
        mi = mutual_info_classif(x_tr, y_tr)
        mi_series = pd.Series(mi, index=x_tr.columns)
        significant = mi_series.sort_values(ascending=False).head(200).index.tolist()

    # --- 6. ANOVA F-test ---
    elif filter_method == "f_test":
        _, p_vals = f_classif(x_tr, y_tr)
        p_series = pd.Series(p_vals, index=x_tr.columns)
        significant = p_series[p_series < p_thresh].index.tolist()

    else:
        raise ValueError(f"Unknown filter method: {filter_method}")

    return significant

# --- CORRECTED grid_search function ---

def grid_search(method_cfg: dict, data: tuple, feature_names=None, k_val: int = 5, refit_metric: str = 'auc_prc') -> Union[dict, pd.DataFrame]:
    """
    Run grid search for any feature selection configuration.
    Builds a pipeline automatically for selectors (SFS, RFE, etc.).
    
    Parameters:
        method_cfg : dict
            Configuration dictionary for the method.
        data : tuple
            Tuple of (x_train, x_test, y_train, y_test).
        feature_names : list, optional
            List of feature names.
        k_val : int, default=5
            Number of folds for cross-validation.
        refit_metric : str, default='accuracy'
            Metric to use for refitting the best estimator. Options:
            - 'accuracy': Uses accuracy (always available)
            - 'roc_auc': Uses ROC-AUC (requires predict_proba)
            - 'auc_prc': Uses AUC-PRC (requires predict_proba)
            - 'sensitivity': Uses recall/sensitivity (always available)
            Falls back to 'accuracy' if the metric requires predict_proba but estimator doesn't support it.
    """
    x_tr, x_ts, y_tr, y_ts = data
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(x_tr.shape[1])]

    filter_method = method_cfg.get("filter_method", None)
    filtered_feats = get_filtered_features(filter_method, x_tr, y_tr)
    estimator_cls = method_cfg["estimator"]
    
    # Initialize metric variables outside the 'if selected_feats' block
    acc, f1, prec, rec, auc_prc, auc_roc, specificity, sensitivity = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    selected_feats = []


 ## UNPACK AND CORRECTLY ROUTE FITTING PARAMS (CORRECTED)
    fitting_params = {}
    if "fit_params" in method_cfg:
        fitting_cfg = method_cfg.get("fit_params", {})
        sample_weights = fitting_cfg.get("sample_weights")
        if sample_weights is not None:
            # Only pass sample_weight to the final classifier, not to the selector's estimator
            if "feature_selector" in method_cfg and method_cfg["feature_selector"].get("selector"):
                fitting_params["clf__sample_weight"] = sample_weights
            else:
                # If no feature selector, the estimator is the only step
                fitting_params[f"{estimator_cls.__name__}__sample_weight"] = sample_weights
    # ... (rest of the grid_search function continues) ...

    
    # --- Handle methods with feature selectors (Wrapper/Embedded) ---
    if "feature_selector" in method_cfg and method_cfg["feature_selector"].get("selector"):
        selector_cfg = method_cfg["feature_selector"]
        selector_cls = selector_cfg["selector"]
        selector_params = selector_cfg.get("param_grid", {})

    
        # FIX 1: Correctly instantiate the final classifier for the pipeline ('clf').
        # Use default/fixed parameters (like random_state), not the lists of search parameters.
        try:
            base_estimator = estimator_cls(random_state=42)

        except TypeError:
            print(f"Cannot set random state for {estimator_cls.__name__}. Using default constructor.")
            base_estimator = estimator_cls()
            
        if estimator_cls == LogisticRegression:
            base_estimator.set_params(max_iter=5000)
        
        # FIX 2: Correctly extract the fixed estimator for the selector (it's in a list in param_grid).
        fs_base_estimator = selector_params.get("estimator", [None])[0]
        selector_init_kwargs = {}

        if fs_base_estimator is not None:
            # SFS, RFE, SelectFromModel require an estimator
            selector_init_kwargs["estimator"] = fs_base_estimator
            selector_instance = selector_cls(**selector_init_kwargs)
        else:
            selector_instance = selector_cls()

        selector_tuple = ('selector', selector_instance)

        pipe = Pipeline([
            selector_tuple,
            ('clf', base_estimator)
        ])

        selector_param_names = selector_instance.get_params().keys()
        selector_param_grid = {}
        passthrough_to_clf = {}
        for key, values in selector_params.items():
            if key == "estimator" or key in selector_param_names:
                selector_param_grid[key] = values
            else:
                passthrough_to_clf[key] = values

        clf_param_grid = method_cfg.get("param_grid", {}).copy()
        clf_param_grid.update(passthrough_to_clf)

        # Adjust parameter grid to match pipeline
        param_grid = {f"selector__{k}": v for k, v in selector_param_grid.items()}
        param_grid.update({f"clf__{k}": v for k, v in clf_param_grid.items()})
        estimator = pipe

    # --- Intrinsic model only (no separate selector, e.g., L1 Logistic Regression) ---
    else:
        # FIX 3: Correctly instantiate the intrinsic model (similar to base_estimator above).
        estimator = estimator_cls(random_state=42)
        if estimator_cls == LogisticRegression:
             estimator.set_params(max_iter=5000)

        param_grid = method_cfg.get("param_grid", {})

    # Check if estimator supports predict_proba for probability-based metrics
    # For pipelines, check the final classifier
    if hasattr(estimator, 'named_steps') and 'clf' in estimator.named_steps:
        final_estimator = estimator.named_steps['clf']
    else:
        final_estimator = estimator
    
    supports_proba = hasattr(final_estimator, 'predict_proba')
    
    # Build scoring dictionary
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "sensitivity": make_scorer(recall_score),
        # "auc_prc" :  make_scorer(average_precision_score, needs_proba=True)  # recall = sensitivity
    }
    
    # Add probability-based metrics only if estimator supports predict_proba
    if supports_proba:
        scoring["roc_auc"] = make_scorer(roc_auc_score, response_method='predict_proba')
        scoring["auc_prc"] = make_scorer(average_precision_score, response_method='predict_proba')
    # Determine refit metric with fallback
    refit_metric_actual = refit_metric
    if refit_metric not in scoring:
        print(f"Warning: Refit metric '{refit_metric}' not available. Available metrics: {list(scoring.keys())}")
        # Try common alternatives
        if refit_metric in ['roc_auc', 'auc_prc'] and not supports_proba:
            print(f"Estimator {type(final_estimator).__name__} does not support predict_proba. Falling back to 'accuracy'.")
            refit_metric_actual = 'accuracy'
        elif refit_metric == 'sensitivity':
            refit_metric_actual = 'sensitivity'
        else:
            refit_metric_actual = 'accuracy'
    
    print(f"Using refit metric: {refit_metric_actual}")

    try:
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=k_val),
            scoring=scoring,
            refit=refit_metric_actual,
            n_jobs=-1,
            verbose=1,
            error_score='raise'
        )

        gs.fit(x_tr[filtered_feats], y_tr,  **fitting_params)
        
    except (ValueError, TypeError) as e:
        # If refit metric fails, fall back to accuracy
        if refit_metric_actual != 'accuracy':
            print(f"Warning: Refit metric '{refit_metric_actual}' failed: {e}")
            print("Falling back to 'accuracy' for refitting...")
            refit_metric_actual = 'accuracy'
            
            gs = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=k_val),
                scoring=scoring,
                refit=refit_metric_actual,
                n_jobs=-1,
                verbose=1,
                error_score='raise'
            )
            gs.fit(x_tr[filtered_feats], y_tr, **fitting_params)
        else:
            raise

    print(f"\n=== {method_cfg['name']} ===")
    print("Best Params:", gs.best_params_)
    print("Best Accuracy:", gs.best_score_)
    selected_feats = get_selected_features(gs.best_estimator_, filtered_feats)

    # Use the best_estimator from GridSearch for final evaluation, as it has the optimal params
    # GridSearchCV with refit=True already refits the best estimator on the full training data
    best_estimator = gs.best_estimator_

    

    if selected_feats:
        print(f"Selected {len(selected_feats)} features.")
        
        # If best_estimator is a Pipeline with a selector, it expects the full filtered feature set
        # The selector will handle feature selection internally
        # If it's just a classifier, use only selected features
        if hasattr(best_estimator, 'named_steps') and 'selector' in best_estimator.named_steps:
            # Pipeline with selector: use full filtered features (selector handles selection)
            # No need to re-fit, GridSearchCV already did that
            y_pred = best_estimator.predict(x_tr[filtered_feats])
            y_pred_test = best_estimator.predict(x_ts[filtered_feats])
        else:
            # Just a classifier: use selected features and re-fit
            best_estimator.fit(x_tr[selected_feats], y_tr)
            y_pred = best_estimator.predict(x_tr[selected_feats])
            y_pred_test = best_estimator.predict(x_ts[selected_feats])
        
        # Use test predictions for evaluation
        y_pred = y_pred_test
        
        # Check if the best estimator supports predict_proba
        if hasattr(best_estimator, 'predict_proba'):
            # Use same feature set as for prediction
            if hasattr(best_estimator, 'named_steps') and 'selector' in best_estimator.named_steps:
                y_prob = best_estimator.predict_proba(x_ts[filtered_feats])[:, 1]
            else:
                y_prob = best_estimator.predict_proba(x_ts[selected_feats])[:, 1]
            auc_prc = average_precision_score(y_ts, y_prob)
            auc_roc = roc_auc_score(y_ts, y_prob)


            curve_output_dir = os.path.join(
                os.getcwd(), "graphs", "roc_auc"
            )

                
            plot_roc_curve(
                y_true= y_ts,
                y_prob = y_prob, 
                output_dir = curve_output_dir, 
                filename = method_cfg["name"]
            )
                    
            plot_pr_curve(
                y_true= y_ts,
                y_prob = y_prob, 
                output_dir = curve_output_dir, 
                filename = method_cfg["name"]
            )

        else:
            print("predict_proba not supported, skipping graph plotting")
            y_prob = None
            auc_prc = 0.0
            auc_roc = 0.0

        # --- CORE METRICS ---
        acc = accuracy_score(y_ts, y_pred)
        f1 = f1_score(y_ts, y_pred)
        prec = precision_score(y_ts, y_pred)
        rec = recall_score(y_ts, y_pred)  # sensitivity

        # --- CONFUSION MATRIX FOR SPECIFICITY ---
        tn, fp, fn, tp = confusion_matrix(y_ts, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = rec  # identical to recall

        # --- PRINT EVERYTHING ---
        print("\n--- Test Set Metrics ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"AUC-PRC: {auc_prc:.4f}")

    else:
        # If no features were explicitly selected (e.g., L1 intrinsic model), 
        # use all filtered features for final evaluation.
        print(f"No explicit feature subset detected (intrinsic model). Using all {len(filtered_feats)} filtered features.")
        
        best_estimator.fit(x_tr[filtered_feats], y_tr) 
        y_pred = best_estimator.predict(x_ts[filtered_feats])
        
        if hasattr(best_estimator, 'predict_proba'):
            y_prob = best_estimator.predict_proba(x_ts[filtered_feats])[:, 1]
            auc_prc = average_precision_score(y_ts, y_prob)
            auc_roc = roc_auc_score(y_ts, y_prob)

        # Calculate metrics using all filtered features
        acc = accuracy_score(y_ts, y_pred)
        f1 = f1_score(y_ts, y_pred)
        prec = precision_score(y_ts, y_pred)
        rec = recall_score(y_ts, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_ts, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = rec
        
        print("\n--- Test Set Metrics (Using All Filtered Features) ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        # ... (rest of prints omitted for brevity)
        
        # Set selected_feats to filtered_feats for record keeping
        selected_feats = filtered_feats


    res = {
        "name": method_cfg["name"],
        "filter_method": filter_method,
        "estimator_name": estimator_cls.__name__,
        # Storing best_estimator and cv_results is likely causing issues 
        # if you try to save the DataFrame to Excel without special handling.
        # Removing "best_estimator" and "cv_results" from the final result dictionary for now, 
        # as they are complex objects that Pandas ExcelWriter can't handle directly.
        "best_params": gs.best_params_,
        "best_validation_accuracy": gs.best_score_,
        "test_accuracy": acc,
        "test_f1": f1,
        "test_precision": prec,
        "test_recall": rec,
        "test_sensitivity": sensitivity,
        "test_specificity": specificity,
        "test_auc_roc": auc_roc,
        "test_auc_prc": auc_prc,
        "selected_features": selected_feats,
    }

    # Handle the feature selector name for logging and results table
    fs_name = method_cfg.get("feature_selector", {}).get("selector", estimator_cls).__name__
    fs_alias = method_cfg.get("feature_selector", {}).get("alias", None)
    res["feature_selector"] = fs_alias if fs_alias else fs_name
    
    # Store the param grid used for GridSearchCV (useful for debugging)
    res["grid_search_params"] = str(param_grid) 
    
    # Create the result DataFrame
    res_for_pd_dict = {key: [res[key]] for key in res}
    result_df = pd.DataFrame.from_dict(res_for_pd_dict)
    
    # Store the full CV results (can be large)
    res["cv_results"] = pd.DataFrame(gs.cv_results_)

    return res, result_df