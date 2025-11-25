import os
import pandas as pd
import numpy as np
import math
import sklearn
from classes.preprocessing_classes import CleanFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Union, Literal, TypeAlias
from sklearn.model_selection import train_test_split

#==PIPELINE EXAMPLE==#
 # preprocess_pipe = Pipeline(steps=[
    #     ('screening', CleanFeatureSelector()),
    #     ('impute', SimpleImputer(strategy='mean')),
    #     # ('screening', MulticollinearityRemover(threshold=30)),
    #     ('scale', StandardScaler())
    # ]) 

def train_test_split_patients(
    dataframe: pd.DataFrame, 
    identifier: str, 
    endpoint: str, 
    test_ratio: float = 0.3,
    random_state: int = 42
):
    # Get unique patient IDs
    unique_patients = dataframe[identifier].unique()
    
    # Split patients into train and test
    train_patients, test_patients = train_test_split(
        unique_patients, 
        test_size=test_ratio,
        random_state=random_state,
        stratify=dataframe.groupby(identifier)[endpoint].first()  # Optional: maintain class balance
    )

    df_cpy = dataframe.copy().reset_index()
    
    # Create masks
    train_mask = df_cpy[identifier].isin(train_patients)
    test_mask = df_cpy[identifier].isin(test_patients)

    # Split the data
    x_train = df_cpy[train_mask].drop(columns=[endpoint, identifier], axis=1)
    x_test = df_cpy[test_mask].drop(columns=[endpoint, identifier], axis=1)
    y_train = df_cpy[train_mask][endpoint]
    y_test = df_cpy[test_mask][endpoint]

 
    return x_train, x_test, y_train, y_test


def get_sample_weights_for_split(sample_weights, 
    train_mask) -> tuple[np.ndarray, np.ndarray]:
    return sample_weights[train_mask]
    
Processed: TypeAlias = Union[tuple[pd.DataFrame, pd.DataFrame], tuple[np.ndarray, np.ndarray]]
def preprocess_data_w_pipeline(input_data : tuple[pd.DataFrame, pd.DataFrame], 
                               preprocess_pipe : sklearn.pipeline.Pipeline,
                               keep_cols=True,
                               output : Union[Literal["dataframe"], Literal["ndarray"]]="dataframe") -> Processed:
    if not isinstance(preprocess_pipe, sklearn.pipeline.Pipeline):
        raise TypeError("preprocess_pipe argument must be instance if sklearn.pipeline.Pipeline!")

    if output not in ["dataframe", "ndarray"]:
        raise ValueError("output must either be 'dataframe' or 'ndarray'!")

    if not isinstance(input_data, tuple) or len(input_data) != 2:
        raise TypeError("input_data must be a tuple of (x_train, x_test) DataFrames.")

    x_tr, x_ts = input_data
    original_x_tr, original_x_ts = x_tr, x_ts

    try:
        x_tr_index = x_tr.index
        x_ts_index = x_ts.index
        x_cols = x_tr.columns
    
        x_tr = preprocess_pipe.fit_transform(x_tr)
        x_ts = preprocess_pipe.transform(x_ts)
        # rebuild DataFrames with only the kept columns
        kept_cols = preprocess_pipe.named_steps['screening'].keep_features_
        # print(kept_cols)
        x_tr = pd.DataFrame(x_tr, columns=kept_cols, index=x_tr_index)
        x_ts = pd.DataFrame(x_ts, columns=kept_cols, index=x_ts_index)

        if output == "ndarray":
            return x_tr.to_numpy(), x_ts.to_numpy()

        return x_tr, x_ts
    
    except Exception as e:
        print(e)
        return original_x_tr, original_x_ts
