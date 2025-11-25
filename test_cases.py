from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, accuracy_score
from sklearn.feature_selection import RFE, SelectKBest, SequentialFeatureSelector, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
# from sklearn.unsupervised.KMeans import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectFromModel
from xgboost import XGBClassifier
# NOTE: These configurations exclusively use Wrapper (RFE, SFS) and Embedded (SFM, L1 Regularization) methods.
# Filter methods (like SelectKBest) have been removed.

# --- 1. CONFIGURATIONS FOR LINEAR MODELS (Logistic Regression) ---

fs_methods_lr = [
    # 1. WRAPPER METHOD: Sequential Forward Selection (SFS)
    {
        "name": "mannwhitney_SFS_LR",
        "estimator": LogisticRegression,
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": SequentialFeatureSelector,
            "param_grid": {
                "estimator": [LogisticRegression(solver='liblinear', random_state=42)],
                "n_features_to_select": [10, 20, 30, 40],
                "direction": ["forward"],
                "cv": [3]
            }
        },
        "param_grid": {
            "C": [0.5, 1.0, 2.0, 5.0],
            "solver": ['liblinear']
        }
    },
    
    # 2. WRAPPER METHOD: Recursive Feature Elimination (RFE)
    {
        "name": "mannwhitney_RFE_LR",
        "estimator": LogisticRegression,
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": RFE,
            "param_grid": {
                "estimator": [LogisticRegression(solver='liblinear', random_state=42)],
                "n_features_to_select": [10, 20, 30, 40],
                "step": [5]
            }
        },
        "param_grid": {
            "C": [0.5, 5.0],
            "solver": ['liblinear']
        }
    },

    # 3. EMBEDDED METHOD: Select From Model (SFM)
    {
        "name": "mannwhitney_LASSO_LR",
        "estimator": LogisticRegression,
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": SelectFromModel,
            "alias" : "LASSO",
            "param_grid": {
                # Uses a sparse linear model (L1/Lasso) to select features based on non-zero coefficients
                "estimator": [LinearSVC(C=0.1, penalty='l1', dual=False, random_state=42)],
                "threshold": ["1.25*mean", "median"]
            }
        },
        "param_grid": {
            "C": [0.1, 1.0, 10.0],
            "solver": ['liblinear']
        }
    },
    
    # 4. EMBEDDED METHOD: Lasso (L1) Regularization directly on the model
    {
        "name": "mannwhitney_L1_LogisticRegression",
        "estimator": LogisticRegression,
        "filter_method": "mannwhitney",
        "feature_selector": {}, # The model itself performs the feature selection via L1 penalty
        "param_grid": {
            "C": [0.01, 0.1, 1.0, 10.0], # Tuning regularization strength is key here
            "penalty": ['l1'],
            "solver": ['liblinear'] # Required solver for L1
        }
    }
]

# --- 2. CONFIGURATIONS FOR TREE-BASED MODELS (Random Forest) ---

fs_methods_rf = [
    # 5. WRAPPER METHOD: Sequential Forward Selection (SFS)
    {
        "name": "mannwhitney_SFS_RF",
        "estimator": RandomForestClassifier,
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": SequentialFeatureSelector,
            "param_grid": {
                "estimator": [LogisticRegression(solver='liblinear', random_state=42)], 
                "n_features_to_select": [30, 60],
                "direction": ["forward"],
                "cv": [3]
            }
        },
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [15],
            "min_samples_split": [5]
        }
    },
    
    # 6. WRAPPER METHOD: Recursive Feature Elimination (RFE)
    {
        "name": "mannwhitney_RFE_RF",
        "estimator": RandomForestClassifier,
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": RFE,
            "param_grid": {
                "estimator": [RandomForestClassifier(n_estimators=50, random_state=42)],
                "n_features_to_select": [50, 80],
                "step": [5]
            }
        },
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [15],
            "min_samples_split": [5]
        }
    },

    # 7. EMBEDDED METHOD: Select From Model (SFM)
    {
        "name": "mannwhitney_SelectFromModel_RF",
        "estimator": RandomForestClassifier,
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": SelectFromModel,
            "param_grid": {
                # Uses feature importances from a fitted RF model
                "estimator": [RandomForestClassifier(n_estimators=100, random_state=42)],
                "threshold": ["1.25*mean", "median"]
            }
        },
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [15],
            "min_samples_split": [5]
        }
    }
]

# --- 3. CONFIGURATIONS FOR BOOSTED MODELS (XGBoost) ---

fs_methods_xgb = [
    # 8. WRAPPER: Sequential Forward Selection (SFS)
    {
        "name": "mannwhitney_SFS_XGB",
        "estimator": XGBClassifier,
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": SequentialFeatureSelector,
            "param_grid": {
                # Using a computationally cheaper estimator (Logistic Regression) for the SFS step
                "estimator": [LogisticRegression(solver='liblinear', random_state=42)], 
                "n_features_to_select": [30, 60, 90], # Expanded search space
                "direction": ["forward"], 
                "cv": [3]
            }
        },
        # Expanded and standardized parameter grid for the final XGBoost model
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],               # New: Ratio of samples used for boosting
            "colsample_bytree": [0.8, 1.0],        # New: Ratio of features used per tree
            "reg_alpha": [0.01, 0.1],              # New: L1 regularization term on weights
            "use_label_encoder": [False],
            "eval_metric": ['logloss']
        }
    },
    
    # 9. WRAPPER: Recursive Feature Elimination (RFE)
    {
        "name": "mannwhitney_RFE_XGB",
        "estimator": XGBClassifier,
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": RFE,
            "param_grid": {
                # Using a smaller XGBoost model for the RFE ranking step for speed
                "estimator": [XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss', max_depth=3)],
                "n_features_to_select": [30, 60, 90], # Expanded search space
                "step": [5, 10]                        # Expanded: Increased step size options
            }
        },
        # Standardized parameter grid for the final XGBoost model
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_alpha": [0.01, 0.1],
            "use_label_encoder": [False],
            "eval_metric": ['logloss']
        }
    },

    # 10. EMBEDDED: Select From Model (SFM)
    {
        "name": "mannwhitney_SelectFromModel_XGB",
        "estimator": XGBClassifier,
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": SelectFromModel,
            "alias" : "XGB_FeatureImportance",
            "param_grid": {
                # Uses feature importances from a fitted XGB model for selection
                "estimator": [XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss', max_depth=3)],
                "threshold": ["1.25*mean", "median", "2*mean"] # Expanded: Added a tighter threshold
            }
        },
        # Standardized parameter grid for the final XGBoost model
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_alpha": [0.01, 0.1],
            "use_label_encoder": [False],
            "eval_metric": ['logloss']
        }
    }
]

# --- 4. CONFIGURATIONS FOR SUPPORT VECTOR MACHINES (SVC and LinearSVC) ---

fs_methods_svm = [
    # 11. WRAPPER: Sequential Forward Selection (SFS)
    {
        "name": "mannwhitney_SFS_SVC",
        "estimator": SVC,
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": SequentialFeatureSelector,
            "param_grid": {
                "estimator": [LinearSVC(C=0.1, dual=False, random_state=42)],
                "n_features_to_select": [10, 20, 30, 40], "direction": ["forward"], "cv": [3]
            }
        },
        "param_grid": {
            "C": [0.1, 1.0, 2.0, 5.0, 10.0, 20.0],  "kernel": ['linear', 'polynomial', 'rbf'],
            "probability": [True]
        }
    },
    
    # 12. WRAPPER: Recursive Feature Elimination (RFE)
    {
        "name": "mannwhitney_RFE_SVC",
        "estimator": SVC,
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": RFE,
            "param_grid": {
                "estimator": [LinearSVC(C=0.1, dual=False, random_state=42)],
                "n_features_to_select": [10, 20, 30], "step": [5]
            }
        },
        "param_grid": {
            "C": [0.1, 1.0, 2.0, 5.0, 10.0, 20.0],  "kernel": ['linear', 'poly', 'rbf'],
            "probability": [True]
        }
    },

    # 13. EMBEDDED: Select From Model (SFM)
    {
        "name": "mannwhitney_SelectFromModel_SVC",
        "estimator": SVC,
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": SelectFromModel,
            "param_grid": {
                "estimator": [LinearSVC(C=1.0, penalty='l1', dual=False, random_state=42)],
                "threshold": ["1.25*mean", "median"]
            }
        },
        "param_grid": {
            "C": [0.1, 1.0, 2.0, 5.0, 10.0, 20.0], "kernel": ['linear', 'poly', 'rbf'],
            "probability": [True]
        }
    },
    
    # 14. EMBEDDED METHOD: Lasso (L1) Regularization directly on the LinearSVC model
    # Note: We must use LinearSVC (instead of SVC) for L1 penalty support.
    {
        "name": "mannwhitney_L1_LinearSVC",
        "estimator": LinearSVC, 
        "filter_method": "mannwhitney",
        # "feature_selector": None, 
        "param_grid": {
            "C": [0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
            "penalty": ['l1'],
            "dual": [False], # Must be False for L1 penalty
            "random_state": [42]
        }
    }, 

    {
        "name": "mannwhitney_PCA_LinearSVC",
        "estimator": LinearSVC, 
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": PCA,

            "param_grid": {
                "n_components": [10, 20, 30]
            }
        }, 
        "param_grid": {
            "C": [0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
            "penalty": ['l1'],
            "dual": [False], # Must be False for L1 penalty
            "random_state": [42]
        }
    }
    
]


fs_methods_knn = [
    # 20. WRAPPER: Sequential Forward Selection (SFS) with KNN
    # SFS works well with KNN because it tests features based on actual performance improvement.
    {
        "name": "mannwhitney_SFS_KNN",
        "estimator": KNeighborsClassifier,
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": SequentialFeatureSelector,
            "param_grid": {
                # Using KNN itself inside the selector to optimize specifically for neighbor distance
                "estimator": [KNeighborsClassifier(n_neighbors=5)], 
                "n_features_to_select": [10, 20, 30], 
                "direction": ["forward", "backward"], 
                "cv": [3]
            }
        },
        "param_grid": {
            "n_neighbors": [3, 5, 7, 9], # Generally use odd numbers to avoid ties
            "weights": ['uniform', 'distance'], # 'distance' weighs closer neighbors more heavily
            "metric": ['euclidean', 'manhattan', 'minkowski'],
            "p": [1, 2] # Power parameter for Minkowski (1=Manhattan, 2=Euclidean)
        }
    },

    # 21. WRAPPER: Recursive Feature Elimination (RFE)
    # CRITICAL NOTE: KNN cannot be used *inside* RFE because it has no 'coef_' or 'feature_importances_'.
    # # We must use a surrogate model (like Random Forest) to rank the features, then feed them to KNN.
    # {
    #     "name": "mannwhitney_RFE_KNN",
    #     "estimator": KNeighborsClassifier,
    #     "filter_method": "mannwhitney",
    #     "feature_selector": {
    #         "selector": RFE,
    #         "param_grid": {
    #             # We use RF to find the "best" features, then use KNN to classify them.
    #             "estimator": [RandomForestClassifier(n_estimators=50, random_state=42)],
    #             "n_features_to_select": [10, 20, 30],
    #             "step": [5]
    #         }
    #     },
    #     "param_grid": {
    #         "n_neighbors": [5, 7, 9],
    #         "weights": ['uniform', 'distance'],
    #         "metric": ['euclidean', 'manhattan']
    #     }
    # },

    # 22. METRIC LEARNING: Neighborhood Components Analysis (NCA)
    # This is the "Gold Standard" feature extraction for KNN.
    # It learns a linear transformation of the data specifically to maximize KNN accuracy.
    # It is strictly supervised (uses y_train).
    {
        "name": "mannwhitney_NCA_KNN",
        "estimator": KNeighborsClassifier,
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": NeighborhoodComponentsAnalysis,
            "param_grid": {
                # n_components is the number of dimensions to reduce to
                "n_components": [10, 20, 30], 
                "init": ['auto', 'pca', 'lda'], # Initialization method
                # "random_state": [42]
            }
        },
        "param_grid": {
            "n_neighbors": [5, 7, 9, 11],
            "weights": ['uniform', 'distance'],
            "metric": ['euclidean'] # NCA works best with Euclidean distance in the transformed space
        }
    }
]

# fs_methods_svm = [fs_methods_svm[-1]]
# Combine all configurations for easy import in the main script

fs_methods_knn = fs_methods_knn[0]
fs_methods_all = (fs_methods_lr, fs_methods_rf, fs_methods_xgb, fs_methods_svm, fs_methods_knn)