
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
from custom_models.svm_shap import SVMSHAP
# At the top with other imports

# In your test case configuration
# {
#     "name": "SFS_LR",
#     "estimator": LogisticRegression,
#     "filter_method": "mannwhitney",
#     "feature_selector": {
#         "selector": SequentialFeatureSelector,
#         "param_grid": {
#             "estimator": [LogisticRegression(solver='liblinear', random_state=42)],
#             "n_features_to_select": [10, 20],
#             "direction": ["forward", "backward"],
#             "cv": [3]
#         }
#     },
#     "param_grid": {
#         "C": [0.5],
#         "solver": ['liblinear']
#     }
#     # Note: No fit_params here anymore
# }
fs_methods_lr = [
    # 1. WRAPPER METHOD: Sequential Forward Selection (SFS)
    {
        "name": "SFS_LR",
        "estimator": LogisticRegression,
        "filter_method": "mannwhitney",
        "feature_selector": {
            "selector": SequentialFeatureSelector,
            "param_grid": {
                "estimator": [LogisticRegression(solver='liblinear', random_state=42)],
                "n_features_to_select": [10, 20, 30],
                "direction": ["forward", "backward"],
                "cv": [3]
            }
        },


        "fit_params" : {
            "sample_weight" : "balanced"
        },
        
        "param_grid": {
            "C": [1.0],
            "solver": ['liblinear']
        }
    },
    
    # 2. WRAPPER METHOD: Recursive Feature Elimination (RFE)
    

    # 3. EMBEDDED METHOD: Select From Model (SFM)
    {
        "name": "LASSO_LR",
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
        # "fit_params" : {
        #     "sample_weights" : sample_weights
        # },
        "param_grid": {
            "C": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
            "solver": ['liblinear']
        }
    },
    
    # 4. EMBEDDED METHOD: Lasso (L1) Regularization directly on the model
    {
        "name": "L1_LogisticRegression",
        "estimator": LogisticRegression,
        "filter_method": "mannwhitney",
        "feature_selector": {}, # The model itself performs the feature selection via L1 penalty
        "param_grid": {
            "C": [0.01 , 0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0], # Tuning regularization strength is key here
            "penalty": ['l1'],
            "solver": ['liblinear'] # Required solver for L1
        }
    }
]


# fs_methods_lr = [fs_methods_lr[0]]

from xgboost import XGBClassifier
from sklearn.feature_selection import SequentialFeatureSelector, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from sklearn.feature_selection import SequentialFeatureSelector, RFE, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # New import for non-linear SFS

fs_methods_xgb = [
    # 8. WRAPPER: Sequential Forward Selection (SFS)
    {
        "name": "NonLinear_SFS_XGB",
        "estimator": XGBClassifier,
        "filter_method": None,
        "feature_selector": {
            "selector": SequentialFeatureSelector,
            "param_grid": {
                "estimator": [RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)], 
                "n_features_to_select": [5, 10, 20], 
                "direction": ["forward"], 
                "cv": [3]
            }       
        },

        "fit_params" : {
            "sample_weight" : "balanced"
        },
    
        "param_grid": {
            # ... XGBoost params
        }
    },
    
    # 9. WRAPPER: Recursive Feature Elimination (RFE)
    {
        "name": "RFE_XGB",
        "estimator": XGBClassifier,
        "filter_method": None,
        "feature_selector": {
            "selector": RFE,
            "param_grid": {
                # CORRECT: Tuning XGBoost parameters by creating a list of different estimators
                "estimator": [
                    XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                ],
                "n_estimators" : [100, 200, 300 , 500],
                "max_depth" : [3,5,10,20], #  Expanded search space
                "n_features_to_select": [30, 60, 90], 
                "step": [5, 10] 
            }
        },
        # "fit_params" : {
        #     "sample_weights" : sample_weights
        # }, 
        "param_grid": {
            # ... XGBoost params
        }
    },

    # 10. EMBEDDED: Select From Model (SFM)
    {
        "name": "SFM_XGB",
        "estimator": XGBClassifier,
        "filter_method": None,
        "feature_selector": {
            "selector": SelectFromModel,
            "alias" : "XGB_FeatureImportance",
            "param_grid": {
                # CORRECT: Tuning XGBoost parameters by creating a list of different estimators
                "estimator": [
                    XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                ],
                "n_estimators" : [100, 200, 300 , 500],
                "max_depth" : [3,5,10, 20], #  Expanded search space
                # FIX: Added comma below and fixed parameter list
                "threshold": ["1.25*mean", "median", "2*mean"] 
            }
        },
        # "fit_params" : {
        #     "sample_weights" : sample_weights
        # },
        "param_grid": {
            # ... XGBoost params
        } 
    },

    # 11. FILTER: Select K Best using Mutual Information (Non-linear Filter)
    # {
    #     "name": "MI_Filter_XGB",
    #     "estimator": XGBClassifier,
    #     "filter_method": {
    #         "selector": SelectKBest,
    #         "param_grid": {
    #             # 🚀 FIX: Mutual Information detects non-linear relationships
    #             "score_func": [mutual_info_classif],
    #             "k": [60, 90, 120] # Keep a large number of features for the model
    #         }
    #     },
    #     "feature_selector": {}, # No further selection
    #     "param_grid": {
    #         # ... XGBoost params
    #     }
    # }
]


fs_methods_svm = [
    # {
    #     "name": "SFS_forward_SVC",
    #     "estimator": SVMSHAP,
    #     "filter_method": None,
    #     "feature_selector": {
    #         "selector": SequentialFeatureSelector,
    #         "param_grid": {
    #             "estimator": [SVMSHAP(C=1.0, random_state=42)],
    #             "n_features_to_select": [5, 10],
    #             "direction": ["forward"],
    #             "cv": [3]
    #         }
    #     }
    # },
      {
        "name": "SFS_forward_SVC_2",
        "estimator": SVMSHAP,
        "filter_method": None,
        "feature_selector": {
            "selector": SequentialFeatureSelector,
            "param_grid": {
                "estimator": [SVMSHAP(C=1.0, random_state=42)],
                "n_features_to_select": [15, 20],
                "direction": ["forward"],
                "cv": [3]
            }
        }
    }
]