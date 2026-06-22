"""Training pipeline for DiagnostiX AI.

Trains a multi-output diagnostic model from the device type, brand, usage
details, and three reported symptoms. For a given input it can predict:

* ``Faulty_Component``           - the most likely broken hardware part
* ``Severity``                   - how serious the fault is (Low/Medium/High)
* ``Estimated_Repair_Cost_INR``  - an estimated repair cost in rupees
* ``Estimated_Repair_Time_Hours``- an estimated repair time in hours

The recommended fix is not learned separately: every component maps to exactly
one recommended solution in the dataset, so it is looked up from the predicted
component (see ``component_to_solution`` in the saved artifact).

Run locally with::

    python train.py

The Streamlit app can also train the model automatically on first launch (see
``diagnosis.get_artifact``), so committing ``model.pkl`` is optional.
"""
from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
ARTIFACT_VERSION = 4

# When present, this dataset is always preferred. Other CSVs are only used as a
# fallback (and only if they contain every required column).
PREFERRED_DATASET = "DiagnostiX_AI_2500_Dataset.csv"

# Targets the model learns to predict.
FAULT_TARGET = "Faulty_Component"
SEVERITY_TARGET = "Severity"
COST_TARGET = "Estimated_Repair_Cost_INR"
TIME_TARGET = "Estimated_Repair_Time_Hours"
# Looked up from the predicted component (1:1 with Faulty_Component).
SOLUTION_COLUMN = "Recommended_Solution"

FEATURE_COLUMNS = [
    "Device",
    "Brand",
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
    "Brand",
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

REQUIRED_COLUMNS = FEATURE_COLUMNS + [
    FAULT_TARGET,
    SEVERITY_TARGET,
    COST_TARGET,
    TIME_TARGET,
    SOLUTION_COLUMN,
]


def resolve_data_path():
    """Locate the training dataset.

    The provided ``DiagnostiX_AI_2500_Dataset.csv`` is preferred whenever it is
    present. Otherwise the largest CSV in the project folder that contains
    every required column is used, so the app still works if the file is
    renamed or re-uploaded.
    """
    preferred = BASE_DIR / PREFERRED_DATASET
    if preferred.exists():
        try:
            header = pd.read_csv(preferred, nrows=1)
            if set(REQUIRED_COLUMNS).issubset(header.columns):
                return preferred
        except Exception:
            pass

    valid_datasets = []
    for candidate in sorted(BASE_DIR.glob("*.csv")):
        try:
            header = pd.read_csv(candidate, nrows=1)
        except Exception:
            continue
        if set(REQUIRED_COLUMNS).issubset(header.columns):
            valid_datasets.append(candidate)

    if not valid_datasets:
        raise FileNotFoundError(
            "No dataset CSV found. Add a CSV that contains the columns: "
            + ", ".join(REQUIRED_COLUMNS)
        )

    return max(valid_datasets, key=lambda path: path.stat().st_size)


def load_dataset(path=None):
    path = Path(path) if path else resolve_data_path()
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(sorted(missing))
        )
    return df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def build_classifier_pipeline():
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_regressor_pipeline():
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_component_to_solution(df):
    """Map each faulty component to its recommended solution (1:1 in the data)."""
    mapping = {}
    for component, group in df.groupby(FAULT_TARGET):
        mapping[str(component)] = str(group[SOLUTION_COLUMN].mode().iloc[0])
    return mapping


def build_input_options(df):
    symptom_combinations_by_device = {}
    brands_by_device = {}
    for device in sorted(df["Device"].unique()):
        device_rows = df[df["Device"] == device]
        combinations = device_rows[
            ["Symptom1", "Symptom2", "Symptom3"]
        ].drop_duplicates()
        symptom_combinations_by_device[device] = combinations.to_dict(
            orient="records"
        )
        brands_by_device[device] = sorted(device_rows["Brand"].unique().tolist())
    return {
        "devices": sorted(df["Device"].unique().tolist()),
        "brands": sorted(df["Brand"].unique().tolist()),
        "brands_by_device": brands_by_device,
        "usage_types": sorted(df["Usage_Type"].unique().tolist()),
        "symptom_combinations_by_device": symptom_combinations_by_device,
    }


def train_and_save(data_path=None, model_path=MODEL_PATH, evaluate=False):
    """Train all predictors and return an artifact dict.

    When ``evaluate`` is False (the default, used for fast in-app training)
    only the final models are fitted. When True, a held-out test split is used
    to compute honest metrics for every target.
    """
    df = load_dataset(data_path)
    X = df[FEATURE_COLUMNS]
    y_fault = df[FAULT_TARGET]
    y_severity = df[SEVERITY_TARGET]
    y_cost = df[COST_TARGET]
    y_time = df[TIME_TARGET]

    metrics = {
        "training_rows": int(len(df)),
        "n_components": int(y_fault.nunique()),
        "n_devices": int(df["Device"].nunique()),
        "n_brands": int(df["Brand"].nunique()),
    }
    fault_report = None

    if evaluate:
        idx_train, idx_test = train_test_split(
            df.index, test_size=0.2, random_state=42, stratify=y_fault
        )
        X_train, X_test = X.loc[idx_train], X.loc[idx_test]

        fault_eval = build_classifier_pipeline()
        fault_eval.fit(X_train, y_fault.loc[idx_train])
        fault_pred = fault_eval.predict(X_test)
        metrics["fault_accuracy"] = float(
            (fault_pred == y_fault.loc[idx_test].values).mean()
        )
        metrics["fault_macro_f1"] = float(
            f1_score(
                y_fault.loc[idx_test], fault_pred, average="macro", zero_division=0
            )
        )
        fault_report = classification_report(
            y_fault.loc[idx_test], fault_pred, zero_division=0, output_dict=True
        )

        sev_eval = build_classifier_pipeline()
        sev_eval.fit(X_train, y_severity.loc[idx_train])
        sev_pred = sev_eval.predict(X_test)
        metrics["severity_accuracy"] = float(
            (sev_pred == y_severity.loc[idx_test].values).mean()
        )

        cost_eval = build_regressor_pipeline()
        cost_eval.fit(X_train, y_cost.loc[idx_train])
        cost_pred = cost_eval.predict(X_test)
        metrics["cost_mae_inr"] = float(
            mean_absolute_error(y_cost.loc[idx_test], cost_pred)
        )
        metrics["cost_r2"] = float(r2_score(y_cost.loc[idx_test], cost_pred))

        time_eval = build_regressor_pipeline()
        time_eval.fit(X_train, y_time.loc[idx_train])
        time_pred = time_eval.predict(X_test)
        metrics["time_mae_hours"] = float(
            mean_absolute_error(y_time.loc[idx_test], time_pred)
        )
        metrics["time_r2"] = float(r2_score(y_time.loc[idx_test], time_pred))

    fault_model = build_classifier_pipeline().fit(X, y_fault)
    severity_model = build_classifier_pipeline().fit(X, y_severity)
    cost_model = build_regressor_pipeline().fit(X, y_cost)
    time_model = build_regressor_pipeline().fit(X, y_time)

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "fault_model": fault_model,
        "severity_model": severity_model,
        "cost_model": cost_model,
        "time_model": time_model,
        "component_to_solution": build_component_to_solution(df),
        "input_options": build_input_options(df),
        "metrics": metrics,
        "fault_classification_report": fault_report,
    }

    try:
        with Path(model_path).open("wb") as model_file:
            pickle.dump(artifact, model_file)
    except OSError:
        pass

    return artifact


if __name__ == "__main__":
    dataset = load_dataset()
    print(
        f"Loaded {len(dataset):,} rows | "
        f"{dataset[FAULT_TARGET].nunique()} components | "
        f"{dataset['Brand'].nunique()} brands"
    )
    artifact = train_and_save(evaluate=True)
    m = artifact["metrics"]
    print(f"Model saved to {MODEL_PATH}")
    print(f"Component accuracy : {m['fault_accuracy']:.2%}")
    print(f"Component macro F1 : {m['fault_macro_f1']:.2%}")
    print(f"Severity accuracy  : {m['severity_accuracy']:.2%}")
    print(
        f"Repair cost  MAE   : Rs {m['cost_mae_inr']:.0f}  (R2 {m['cost_r2']:.2f})"
    )
    print(
        f"Repair time  MAE   : {m['time_mae_hours']:.2f} h (R2 {m['time_r2']:.2f})"
    )
