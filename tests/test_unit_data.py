import numpy as np
import pandas as pd


def test_clean_df_removes_constants_and_infs():
    from src.data_processing import clean_df

    df = pd.DataFrame(
        {"A": [1, 1, 1], "B": [np.inf, 2, 3], "C": ["x", "y", "z"], "target": [0, 1, 0]}
    )

    cleaned = clean_df(df, target_col="target", column_map=None, source_name="test_df")

    # Check constant column A is dropped
    assert "A" not in cleaned.columns

    # Check inf replaced by NaN
    assert cleaned["B"].isna().iloc[0]

    # Check shape
    assert cleaned.shape[1] == 3  # B, C, target
