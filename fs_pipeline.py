from random import sample
import pandas as pd
import numpy as np
import os
import gc
import traceback
from methods import grid_search
from test_cases.combined_test_cases import fs_methods_all
from utils.preprocessing import preprocess_data_w_pipeline
from utils.preprocessing import train_test_split_patients
from sklearn.pipeline import Pipeline
from classes.preprocessing_classes import CleanFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from test_cases import clinical_test_cases, combined_test_cases, radiomic_test_cases
(fs_methods_lr, fs_methods_rf, 
fs_methods_xgb, fs_methods_svm, 
fs_methods_knn) = combined_test_cases.fs_methods_all

from utils.pickler import load_and_reconstruct_sample_weights
from utils.data_testers import is_train_test_difference_significant


# Example run
results = []
# output_excel_path = "fs_results_master.xlsx"

# Make sure combined results are collected
combined_res_df = pd.DataFrame()
curr_fs_method_set = fs_methods_knn


parent_path = os.path.dirname(os.getcwd())
data_df = pd.read_csv(os.path.join(os.getcwd(), "data", "combined_data.csv"))


pipeline_dict = {

    "radiomic" : {
        "data" : {
            "df" : pd.read_csv(os.path.join(os.getcwd(), "data", "radiomics_df_final.csv")),
            "endpoint" : "responder", 
            "identifier" : "TCIA_ID"
        },
        "pipeline" : Pipeline(steps=[
            ('screening', CleanFeatureSelector()),
            ('impute', SimpleImputer(strategy='mean')),
            # ('screening', MulticollinearityRemover(threshold=30)),
            ('scale', StandardScaler())
        ]), 
        "test_cases" : radiomic_test_cases.fs_methods_lr,
        "output_path" : os.path.join(os.getcwd(), "data", "fs_results_radiomics.xlsx") 
    }, 

    "combined" : {
        "data" : {
            "df" : pd.read_csv(os.path.join(os.getcwd(), "data", "combined_data.csv")),
            "endpoint" : "responder", 
            "identifier" : "TCIA_ID"
        },
        "pipeline" : Pipeline(steps=[
            ('screening', CleanFeatureSelector()),
            ('impute', SimpleImputer(strategy='mean')),
            # ('screening', MulticollinearityRemover(threshold=30)),
            ('scale', StandardScaler())
        ]), 
        "test_cases" : combined_test_cases.fs_methods_svm,
        "output_path" : os.path.join(os.getcwd(), "data", "fs_results_combined.xlsx") 
    }, 

    "clinical" : {
        "data" : {
            "df" : pd.read_csv(os.path.join(os.getcwd(), "data", "clinical_df_cleaned.csv")),
            "endpoint" : "responder", 
            "identifier" : "TCIA_ID"
        },        
        "pipeline" : Pipeline(steps=[
            ('scale', StandardScaler())
        ]), 

        "test_cases" : clinical_test_cases.fs_methods_lr,
        "output_path" : os.path.join(os.getcwd(), "data", "fs_results_clinical.xlsx") 

    }
}

current_pipeline_key = "radiomic"
seleciton = pipeline_dict[current_pipeline_key]

x_tr, x_ts, y_tr, y_ts = train_test_split_patients(seleciton["data"]["df"] , 
                                                   seleciton["data"]["identifier"],  
                                                   seleciton["data"]["endpoint"],
                                                   )

x_tr, x_ts = preprocess_data_w_pipeline(
    input_data=(x_tr, x_ts),
    preprocess_pipe=seleciton["pipeline"],
)

significant, p_value = is_train_test_difference_significant(x_tr, x_ts)
print(f"Significant difference between train and test: {significant}, p-value: {p_value:.4f}")

for method in seleciton["test_cases"]:
    try:
        # sample_weights = load_and_reconstruct_sample_weights()
        # x_tr_indices = x_tr.index.to_numpy()
        # train_mask = np.isin(np.arange(0, len(sample_weights)), x_tr_indices)

        # weights_series_full = pd.Series(sample_weights)

        # # 2. Use Pandas .loc to select only the weights that correspond to the x_tr index.
        # #    Pandas automatically handles the alignment and order.
        # weights_for_tr_series = weights_series_full.loc[x_tr.index]

        # # 3. Convert the resulting Series back to a NumPy array for use in XGBoost's fit_params.
        # train_weights_final = weights_for_tr_series.to_numpy()
        # train_weights= get_sample_weights_for_split(
        #     sample_weights=sample_weights,
        #     train_mask=train_mask,
        # )

        # #fit sample weights to estimator.fit()
        # method["fit_params"] = {
        #     "sample_weight" : train_weights
        # }
        out, res_df = grid_search(method, (x_tr, x_ts, y_tr, y_ts), feature_names=x_tr.columns, k_val=3)
        combined_res_df = pd.concat([combined_res_df, res_df], ignore_index=True)
        unique_exp_cols = ["name","estimator_name", "filter_method", "feature_selector"]

        if os.path.exists(seleciton["output_path"]):
            existing_df = pd.read_excel(seleciton["output_path"], engine="openpyxl")
            # 1. Concatenate old results and new results. New results must be the last items.
            # This sets up the 'overwrite' by ensuring the newest version is at the bottom.
            final_df = pd.concat([existing_df, combined_res_df], ignore_index=True)
            # 2. Drop duplicates based on the unique columns, keeping the LAST occurrence (the new result).
            # This drops all old rows that match the unique key of the newly added rows.
            final_df = final_df.drop_duplicates(subset=unique_exp_cols, keep='last')
            
        else:
            # If the file doesn't exist, the new results are the final results.
            final_df = combined_res_df
            print("output file does not exist, creating on path", seleciton["output_path"])
        # --- Save back to Excel ---
        with pd.ExcelWriter(seleciton["output_path"], engine="openpyxl", mode="w") as writer:
            final_df.to_excel(writer, index=False)

        gc.collect()

    except Exception:
        traceback.print_exc()
        continue
# --- Merge with existing Excel results if file exists ---

# if os.path.exists(seleciton["output_path"]):
#     existing_df = pd.read_excel(seleciton["output_path"], engine="openpyxl")
#     # 1. Concatenate old results and new results. New results must be the last items.
#     # This sets up the 'overwrite' by ensuring the newest version is at the bottom.
#     final_df = pd.concat([existing_df, combined_res_df], ignore_index=True)
#     # 2. Drop duplicates based on the unique columns, keeping the LAST occurrence (the new result).
#     # This drops all old rows that match the unique key of the newly added rows.
#     final_df = final_df.drop_duplicates(subset=unique_exp_cols, keep='last')
    
# else:
#     # If the file doesn't exist, the new results are the final results.
#     final_df = combined_res_df
#     print("output file does not exist, creating on path", seleciton["output_path"])
# # --- Save back to Excel ---
# with pd.ExcelWriter(seleciton["output_path"], engine="openpyxl", mode="w") as writer:
#     final_df.to_excel(writer, index=False)

