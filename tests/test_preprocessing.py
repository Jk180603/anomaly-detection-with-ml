import sys
import os

sys.path.append(os.path.abspath("."))

import numpy as np
import pandas as pd
from src.preprocessing import create_sliding_windows


def test_create_sliding_windows():
    df = pd.DataFrame({
        "unit_number": [1, 1, 1, 1, 1],
        "sensor_1": [1, 2, 3, 4, 5],
        "sensor_2": [10, 20, 30, 40, 50],
    })

    sequences = create_sliding_windows(
        df=df,
        feature_columns=["sensor_1", "sensor_2"],
        sequence_length=3,
    )

    assert isinstance(sequences, np.ndarray)
    assert sequences.shape == (3, 3, 2)