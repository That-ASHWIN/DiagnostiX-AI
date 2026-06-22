from pathlib import Path
import pickle
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
EXPECTED_ARTIFACT_VERSION = 4


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
            pass

    from train import train_and_save

    return train_and_save()


def create_feature_row(
    device,
    brand,
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
        "Brand": brand,
        "Age_Months": age_months,
        "Daily_Usage_Hours": daily_usage_hours,
        "Failure_After_Months": failure_after_months,
        "Usage_Type": usage_type,
        "Symptom1": symptom1,
        "Symptom2": symptom2,
        "Symptom3": symptom3,
    }


def split_solution_steps(solution_text):
    """Turn a 'Step 1: ...; Step 2: ...' string into a clean list of steps."""
    if not solution_text:
        return []
    parts = re.split(r";\s*", str(solution_text))
    steps = []
    for part in parts:
        cleaned = re.sub(r"^\s*Step\s*\d+\s*:\s*", "", part).strip()
        if cleaned:
            steps.append(cleaned)
    return steps


def _rank_classes(model, sample, top_k):
    probabilities = model.predict_proba(sample)[0]
    ranked_indexes = probabilities.argsort()[::-1][:top_k]
    return [
        {
            "label": str(model.classes_[index]),
            "confidence": float(probabilities[index]),
        }
        for index in ranked_indexes
    ]


def predict_diagnosis(artifact, features, top_k=3):
    """Predict component, severity, cost, time and the recommended solution."""
    feature_columns = artifact["feature_columns"]
    missing = [column for column in feature_columns if column not in features]
    if missing:
        raise ValueError(f"Missing prediction features: {', '.join(missing)}")

    sample = pd.DataFrame(
        [{column: features[column] for column in feature_columns}],
        columns=feature_columns,
    )

    fault_ranking = _rank_classes(artifact["fault_model"], sample, top_k)
    fault = fault_ranking[0]["label"]

    severity_ranking = _rank_classes(artifact["severity_model"], sample, top_k=3)
    severity = severity_ranking[0]["label"]

    estimated_cost = float(artifact["cost_model"].predict(sample)[0])
    estimated_time = float(artifact["time_model"].predict(sample)[0])

    solution_text = artifact.get("component_to_solution", {}).get(fault, "")

    return {
        "fault": fault,
        "confidence": fault_ranking[0]["confidence"],
        "alternatives": [
            {"fault": item["label"], "confidence": item["confidence"]}
            for item in fault_ranking
        ],
        "severity": severity,
        "severity_confidence": severity_ranking[0]["confidence"],
        "estimated_cost_inr": estimated_cost,
        "estimated_time_hours": estimated_time,
        "solution_text": solution_text,
        "solution_steps": split_solution_steps(solution_text),
    }


# Backwards-compatible alias for older callers.
def predict_fault(artifact, features, top_k=3):
    return predict_diagnosis(artifact, features, top_k=top_k)
