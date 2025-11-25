
import pandas as pd
import math

def train_test_split_patients(dataframe : pd.DataFrame, identifier : str, endpoint : str, test_ratio : float):
    unique_patients = dataframe[identifier].unique()

    test_patients = unique_patients[math.floor(1-test_ratio * len(unique_patients)):]
    train_patients = unique_patients[ : math.floor(1-test_ratio * len(unique_patients))]

    train_mask = dataframe[identifier].isin(train_patients)
    test_mask =  dataframe[identifier].isin(test_patients)

    x_train = dataframe[train_mask].drop(columns=[endpoint, identifier], axis=1)
    x_test = dataframe[test_mask].drop(columns=[endpoint, identifier], axis=1)
    y_train = dataframe[train_mask][endpoint]
    y_test = dataframe[test_mask][endpoint]


    return x_train, x_test, y_train, y_test
