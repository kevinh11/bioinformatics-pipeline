from preprocessing import x_tr, x_ts, y_tr, y_ts
from custom_models.stacking_classifier import StackingClassifier
from utils.preprocessing import train_test_split_patients
import pandas as pd
import numpy as np
import os
from test_cases import combined_test_cases, clinical_test_cases

from test_cases.combined_test_cases import fs_methods_all
from utils.preprocessing import preprocess_data_w_pipeline
from utils.preprocessing import train_test_split_patients
from sklearn.pipeline import Pipeline
from classes.preprocessing_classes import CleanFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipeline_dict = {
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

        "test_cases" : clinical_test_cases.fs_methods_svm,
        "output_path" : os.path.join(os.getcwd(), "data", "fs_results_clinical.xlsx") 

    }
}

current_pipeline_key = "clinical"
seleciton = pipeline_dict[current_pipeline_key]

x_tr, x_ts, y_tr, y_ts = train_test_split_patients(seleciton["data"]["df"] , 
                                                   seleciton["data"]["identifier"],  
                                                   seleciton["data"]["endpoint"])
