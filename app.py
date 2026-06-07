import pandas as pd
import streamlit as st

from diagnosis import MODEL_PATH, create_feature_row, load_artifact, predict_fault


st.set_page_config(
    page_title="DiagnostiX AI",
    page_icon="🛠️",
    layout="wide",
)


@st.cache_resource
def get_artifact(model_modified_at):
    return load_artifact()


def main():
    st.title("DiagnostiX AI")
    st.subheader("Electronic Device Fault Prediction")

    try:
        artifact = get_artifact(MODEL_PATH.stat().st_mtime_ns)
    except Exception as error:
        st.error(f"Could not load the ML model: {error}")
        st.stop()

    options = artifact["input_options"]

    with st.sidebar:
        st.header("Device Info")
        st.metric("Devices Supported", len(options["devices"]))
        st.metric("Training Rows", artifact["metrics"]["training_rows"])
        st.metric(
            "Holdout Accuracy",
            f"{artifact['metrics']['test_accuracy']:.1%}",
        )

    device = st.selectbox("Device", options["devices"])
    symptom_combinations = options["symptom_combinations_by_device"][device]

    details_col, usage_col = st.columns(2)
    with details_col:
        age = st.number_input(
            "Device age (months)",
            min_value=1,
            max_value=240,
            value=24,
        )
        failure_after = st.number_input(
            "Problem started after (months)",
            min_value=1,
            max_value=240,
            value=18,
        )
    with usage_col:
        usage = st.number_input(
            "Daily usage (hours)",
            min_value=1,
            max_value=24,
            value=8,
        )
        usage_type = st.selectbox("Usage type", options["usage_types"])

    st.markdown("#### Symptoms")
    symptom_col1, symptom_col2, symptom_col3 = st.columns(3)
    with symptom_col1:
        symptom1_options = sorted(
            {item["Symptom1"] for item in symptom_combinations}
        )
        symptom1 = st.selectbox("Primary symptom", symptom1_options)

    matching_primary = [
        item for item in symptom_combinations if item["Symptom1"] == symptom1
    ]
    with symptom_col2:
        symptom2_options = sorted(
            {item["Symptom2"] for item in matching_primary}
        )
        symptom2 = st.selectbox("Secondary symptom", symptom2_options)

    matching_secondary = [
        item for item in matching_primary if item["Symptom2"] == symptom2
    ]
    with symptom_col3:
        symptom3_options = sorted(
            {item["Symptom3"] for item in matching_secondary}
        )
        symptom3 = st.selectbox("Additional symptom", symptom3_options)

    if st.button("Predict Fault", type="primary", width="stretch"):
        if failure_after > age:
            st.error("Problem start time cannot be greater than the device age.")
            st.stop()

        features = create_feature_row(
            device=device,
            age_months=age,
            daily_usage_hours=usage,
            failure_after_months=failure_after,
            usage_type=usage_type,
            symptom1=symptom1,
            symptom2=symptom2,
            symptom3=symptom3,
        )
        result = predict_fault(artifact, features)

        st.markdown("### Prediction")
        result_col, confidence_col = st.columns(2)
        result_col.metric("Likely faulty component", result["fault"])
        confidence_col.metric("Model confidence", f"{result['confidence']:.1%}")

        alternatives = pd.DataFrame(
            {
                "Component": [
                    item["fault"] for item in result["alternatives"]
                ],
                "Confidence": [
                    f"{item['confidence']:.1%}"
                    for item in result["alternatives"]
                ],
            }
        )
        st.dataframe(alternatives, hide_index=True, width="stretch")
        st.caption(
            "This is an ML estimate. Confirm hardware faults with a technician."
        )


if __name__ == "__main__":
    main()
