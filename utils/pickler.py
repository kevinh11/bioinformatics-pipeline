
import pickle
import os

#script to load all pickle files
sample_weights = None
sample_weights_location = os.path.join(os.getcwd(), "data", "pickle_files", "sample_weights.pkl")

import pickle
import os
from typing import Dict, List, Optional
import numpy as np

# Initialize as None

def load_and_reconstruct_sample_weights() -> Dict[str, float]:
    """Load sample weights from pickle file."""

    sample_weights = None
    if _sample_weights is None:
        sample_weights_location = os.path.join(os.getcwd(), "data", "pickle_files", "sample_weights.pkl")
        with open(sample_weights_location, 'rb') as file:
            sample_weights_info = pickle.load(file)

        sample_weights = np.ones(sample_weights_info["df_length"])

        for idx in sample_weights_info["indices"]:
            sample_weights[idx] = sample_weights_info["weight"]

    return sample_weights

