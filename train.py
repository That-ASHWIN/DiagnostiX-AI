from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "DiagnostiX_AI_600Plus_Dataset - DiagnostiX_600Rows.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

TARGET = "Faulty_Component"
FEATURE_COLUMNS = [
    "Device",
    "Age_Months",
    "Daily_Usage_Hours",
    "Failure_After_Months",
    "Usage_Type",
    "Symptom1",
    "Symptom2",
    "Symptom3",
]
CATEGORICAL_FEATURES = [
    "Device",
    "Usage_Type",
    "Symptom1",
    "Symptom2",
    "Symptom3",
]
NUMERIC_FEATURES = [
    "Age_Months",
    "Daily_Usage_Hours",
    "Failure_After_Months",
]


def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    required_columns = set(FEATURE_COLUMNS + [TARGET])
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[list(required_columns)].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns")

    return df


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_input_options(df):
    symptom_combinations_by_device = {}
    for device in sorted(df["Device"].unique()):
        device_rows = df[df["Device"] == device]
        combinations = device_rows[
            ["Symptom1", "Symptom2", "Symptom3"]
        ].drop_duplicates()
        symptom_combinations_by_device[device] = combinations.to_dict(
            orient="records"
        )

    return {
        "devices": sorted(df["Device"].unique().tolist()),
        "usage_types": sorted(df["Usage_Type"].unique().tolist()),
        "symptom_combinations_by_device": symptom_combinations_by_device,
    }


def train_and_save(data_path=DATA_PATH, model_path=MODEL_PATH):
    df = load_dataset(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    evaluation_model = build_pipeline()
    evaluation_model.fit(X_train, y_train)
    test_accuracy = evaluation_model.score(X_test, y_test)

    final_model = build_pipeline()
    final_model.fit(X, y)

    artifact = {
        "artifact_version": 2,
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "input_options": build_input_options(df),
        "metrics": {
            "test_accuracy": test_accuracy,
            "training_rows": len(df),
        },
    }

    with model_path.open("wb") as model_file:
        pickle.dump(artifact, model_file)

    return artifact


if __name__ == "__main__":
    trained_artifact = train_and_save()
    accuracy = trained_artifact["metrics"]["test_accuracy"]
    print(f"Model saved to {MODEL_PATH}")
    print(f"Holdout accuracy: {accuracy:.2%}")
