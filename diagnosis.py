from pathlib import Path
import pickle

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
EXPECTED_ARTIFACT_VERSION = 3


def load_artifact(path=MODEL_PATH):
    with Path(path).open("rb") as model_file:
        artifact = pickle.load(model_file)

    if not isinstance(artifact, dict):
        raise ValueError(
            "Legacy model detected. Run `python train.py` to rebuild it."
        )
    if artifact.get("artifact_version") != EXPECTED_ARTIFACT_VERSION:
        raise ValueError("Unsupported model artifact. Run `python train.py`.")

    return artifact


def get_artifact():
    """Return a ready-to-use model artifact.

    Loads the saved ``model.pkl`` when it exists and is compatible. Otherwise
    it trains a fresh model directly from the dataset, so the app works on a
    clean deploy without committing the (large) model file.
    """
    if MODEL_PATH.exists():
        try:
            return load_artifact()
        except Exception:
            # Incompatible or corrupt model - retrain from the dataset below.
            pass

    from train import train_and_save

    return train_and_save()


def create_feature_row(
    device,
    age_months,
    daily_usage_hours,
    failure_after_months,
    usage_type,
    symptom1,
    symptom2,
    symptom3,
):
    return {
        "Device": device,
        "Age_Months": age_months,
        "Daily_Usage_Hours": daily_usage_hours,
        "Failure_After_Months": failure_after_months,
        "Usage_Type": usage_type,
        "Symptom1": symptom1,
        "Symptom2": symptom2,
        "Symptom3": symptom3,
    }


def predict_fault(artifact, features, top_k=3):
    feature_columns = artifact["feature_columns"]
    missing = [column for column in feature_columns if column not in features]
    if missing:
        raise ValueError(f"Missing prediction features: {', '.join(missing)}")

    sample = pd.DataFrame(
        [{column: features[column] for column in feature_columns}],
        columns=feature_columns,
    )
    model = artifact["model"]
    predicted_fault = str(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    ranked_indexes = probabilities.argsort()[::-1][:top_k]
    ranking = [
        {
            "fault": str(model.classes_[index]),
            "confidence": float(probabilities[index]),
        }
        for index in ranked_indexes
    ]

    return {
        "fault": predicted_fault,
        "confidence": ranking[0]["confidence"],
        "alternatives": ranking,
    }
