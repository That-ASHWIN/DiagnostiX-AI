"""Training pipeline for DiagnostiX AI.

Trains a Random Forest model that predicts the most likely faulty hardware
component from the device type, usage details, and three reported symptoms.

Run locally with::

    python train.py

The Streamlit app can also train the model automatically on first launch (see
``diagnosis.get_artifact``), so committing ``model.pkl`` is optional.
"""
from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
ARTIFACT_VERSION = 3

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
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET]


def resolve_data_path():
    """Locate the training dataset.

    Picks the largest CSV in the project folder that contains every required
    column. Choosing the largest valid file means the full dataset is used
    even if it was renamed or re-uploaded (for example with a "(1)" suffix),
    while still falling back to the smaller sample dataset when that is the
    only file present.
    """
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


def train_and_save(
    data_path=None,
    model_path=MODEL_PATH,
    evaluate=False,
    run_cross_validation=False,
):
    """Train the model and return an artifact dict.

    When ``evaluate`` is False (the default, used for fast in-app training)
    only the final model is fitted. When True, a held-out test split and an
    optional 5-fold cross-validation are used to compute honest metrics.
    """
    df = load_dataset(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    metrics = {
        "training_rows": int(len(df)),
        "n_classes": int(y.nunique()),
        "n_devices": int(df["Device"].nunique()),
    }
    report = None

    if evaluate:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        evaluation_model = build_pipeline()
        evaluation_model.fit(X_train, y_train)
        y_pred = evaluation_model.predict(X_test)
        metrics["test_accuracy"] = float((y_pred == y_test.values).mean())
        metrics["macro_f1"] = float(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        )
        report = classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        )
        if run_cross_validation:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(build_pipeline(), X, y, cv=cv, n_jobs=-1)
            metrics["cv_accuracy_mean"] = float(scores.mean())
            metrics["cv_accuracy_std"] = float(scores.std())

    final_model = build_pipeline()
    final_model.fit(X, y)

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "input_options": build_input_options(df),
        "metrics": metrics,
        "classification_report": report,
    }

    try:
        with Path(model_path).open("wb") as model_file:
            pickle.dump(artifact, model_file)
    except OSError:
        # Read-only hosting environment: keep using the in-memory artifact.
        pass

    return artifact


def save_confusion_matrix(df=None, png_path=BASE_DIR / "confusion_matrix.png"):
    """Render a confusion matrix image (local/dev use; needs matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except Exception:
        print("matplotlib not available - skipping confusion matrix image.")
        return

    if df is None:
        df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_pipeline().fit(X_train, y_train)
    fig, ax = plt.subplots(figsize=(20, 20))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, ax=ax, xticks_rotation="vertical", colorbar=False
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    print(f"Confusion matrix saved to {png_path}")


if __name__ == "__main__":
    dataset = load_dataset()
    print(f"Loaded {len(dataset):,} rows | {dataset[TARGET].nunique()} components")
    artifact = train_and_save(evaluate=True, run_cross_validation=True)
    m = artifact["metrics"]
    print(f"Model saved to {MODEL_PATH}")
    print(f"Holdout accuracy  : {m['test_accuracy']:.2%}")
    print(f"Macro F1          : {m['macro_f1']:.2%}")
    if "cv_accuracy_mean" in m:
        print(
            f"5-fold CV accuracy: {m['cv_accuracy_mean']:.2%} "
            f"(+/- {m['cv_accuracy_std']:.2%})"
        )
    save_confusion_matrix(dataset)
