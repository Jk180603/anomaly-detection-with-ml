import os
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


COLUMN_NAMES = (
    ["unit_number", "time_in_cycles"]
    + [f"operational_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


DROP_COLUMNS = [
    "operational_setting_3",
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_raw_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )
    return df


def add_rul_column(df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = df.groupby("unit_number")["time_in_cycles"].max()
    df = df.copy()
    df["max_cycles"] = df["unit_number"].map(max_cycles)
    df["RUL"] = df["max_cycles"] - df["time_in_cycles"]
    df.drop(columns=["max_cycles"], inplace=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=DROP_COLUMNS)
    return df


def create_sliding_windows(
    df: pd.DataFrame,
    feature_columns: list,
    sequence_length: int,
) -> np.ndarray:
    sequences = []

    for unit in df["unit_number"].unique():
        unit_data = df[df["unit_number"] == unit][feature_columns].values

        for start in range(0, len(unit_data) - sequence_length + 1):
            end = start + sequence_length
            sequences.append(unit_data[start:end])

    return np.array(sequences)


def preprocess_pipeline(config_path: str = "config/config.yaml"):
    config = load_config(config_path)

    raw_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]
    sequence_length = config["data"]["sequence_length"]

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = load_raw_data(raw_path)
    df = add_rul_column(df)
    df = clean_data(df)

    feature_columns = [
        col for col in df.columns
        if col not in ["unit_number", "time_in_cycles", "RUL"]
    ]

    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    sequences = create_sliding_windows(
        df=df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
    )

    np.save(os.path.join(processed_dir, "train_sequences.npy"), sequences)
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(feature_columns, "models/feature_columns.joblib")

    print(f"Processed dataframe shape: {df.shape}")
    print(f"Training sequence shape: {sequences.shape}")
    print(f"Number of features: {len(feature_columns)}")

    return sequences


if __name__ == "__main__":
    preprocess_pipeline()