
from typing import Union, TypeAlias
import pandas as pd
import numpy as np
Processed: TypeAlias = Union[tuple[pd.DataFrame, pd.DataFrame], tuple[np.ndarray, np.ndarray]]
