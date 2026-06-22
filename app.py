import pandas as pd
import streamlit as st

from diagnosis import create_feature_row, get_artifact, predict_diagnosis
from report import build_report_pdf


st.set_page_config(
    page_title="DiagnostiX AI",
    page_icon="\U0001F6E0\uFE0F",
    layout="wide",
)


@st.cache_resource(
    show_spinner="Preparing the diagnostic model... (the first load can take a moment)"
)
def load_model():
    return get_artifact()


def match_strength_label(score):
    if score >= 0.75:
        return "Strong match"
    if score >= 0.40:
        return "Moderate match"
    return "Weak match - please review the alternatives below"


def render_result(result, inputs):
    st.markdown("### Suggested diagnosis")
    st.success(f"Most likely faulty component: **{result['fault']}**")

    summary_cols = st.columns(3)
    summary_cols[0].metric("Severity", result["severity"])
    summary_cols[1].metric(
        "Est. repair cost", f"Rs {result['estimated_cost_inr']:,.0f}"
    )
    summary_cols[2].metric(
        "Est. repair time", f"{result['estimated_time_hours']:.1f} hrs"
    )

    strength = result["confidence"]
    st.write(f"**Symptom match strength:** {match_strength_label(strength)}")
    st.progress(min(max(strength, 0.0), 1.0))

    if result.get("solution_steps"):
        st.markdown("**Recommended solution:**")
        for step_number, step in enumerate(result["solution_steps"], start=1):
            st.markdown(f"{step_number}. {step}")
    elif result.get("solution_text"):
        st.markdown("**Recommended solution:**")
        st.write(result["solution_text"])

    st.markdown("**Other components worth checking:**")
    alternatives = pd.DataFrame(
        {
            "Component": [item["fault"] for item in result["alternatives"]],
            "Match strength": [
                f"{item['confidence']:.0%}" for item in result["alternatives"]
            ],
        }
    )
    st.dataframe(alternatives, hide_index=True, use_container_width=True)

    st.warning(
        "This is an AI-based estimate from the symptoms you entered. "
        "Please confirm the actual fault with a physical hardware check or "
        "a qualified technician before replacing any part."
    )

    try:
        pdf_bytes = build_report_pdf(inputs, result)
        st.download_button(
            "\U0001F4C4  Download report (PDF)",
            data=pdf_bytes,
            file_name="DiagnostiX_AI_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as error:
        st.caption(f"PDF report could not be generated: {error}")


def main():
    st.title("DiagnostiX AI")
    st.subheader("Electronic Device Fault Prediction")

    st.info(
        "DiagnostiX AI suggests the **most likely** faulty component, its "
        "severity, and an estimated repair cost and time based on real-world "
        "device usage and repair patterns. It is a decision-support guide - "
        "please **confirm with a physical hardware inspection** before "
        "repairing or replacing any part."
    )

    try:
        artifact = load_model()
    except Exception as error:
        st.error(f"Could not prepare the diagnostic model: {error}")
        st.stop()

    options = artifact["input_options"]
    metrics = artifact.get("metrics", {})

    with st.sidebar:
        st.header("About the model")
        st.caption(
            "Trained on real-world device usage and repair records to recognise "
            "the patterns behind common hardware faults."
        )
        st.metric(
            "Device types covered",
            metrics.get("n_devices", len(options["devices"])),
        )
        if metrics.get("n_brands"):
            st.metric("Brands covered", metrics["n_brands"])
        if metrics.get("n_components"):
            st.metric("Components it can flag", metrics["n_components"])
        if metrics.get("training_rows"):
            st.metric("Training records", f"{metrics['training_rows']:,}")
        if metrics.get("fault_accuracy") is not None:
            st.divider()
            st.caption("Held-out test performance")
            st.metric("Component accuracy", f"{metrics['fault_accuracy']:.0%}")
            if metrics.get("severity_accuracy") is not None:
                st.metric(
                    "Severity accuracy", f"{metrics['severity_accuracy']:.0%}"
                )
        st.divider()
        st.caption(
            "Predictions are guidance based on the symptoms you report - always "
            "verify with a hands-on hardware check."
        )

    device = st.selectbox("Device", options["devices"])
    brand_options = options.get("brands_by_device", {}).get(
        device, options["brands"]
    )
    symptom_combinations = options["symptom_combinations_by_device"][device]

    details_col, usage_col = st.columns(2)
    with details_col:
        brand = st.selectbox("Brand", brand_options)
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

    if st.button("Predict Fault", type="primary", use_container_width=True):
        if failure_after > age:
            st.error(
                "Problem start time cannot be greater than the device age."
            )
            st.stop()

        features = create_feature_row(
            device=device,
            brand=brand,
            age_months=age,
            daily_usage_hours=usage,
            failure_after_months=failure_after,
            usage_type=usage_type,
            symptom1=symptom1,
            symptom2=symptom2,
            symptom3=symptom3,
        )
        st.session_state["diagnosis_result"] = predict_diagnosis(
            artifact, features
        )
        st.session_state["diagnosis_inputs"] = {
            "Device": device,
            "Brand": brand,
            "Device age (months)": age,
            "Daily usage (hours)": usage,
            "Problem started after (months)": failure_after,
            "Usage type": usage_type,
            "Primary symptom": symptom1,
            "Secondary symptom": symptom2,
            "Additional symptom": symptom3,
        }

    result = st.session_state.get("diagnosis_result")
    inputs = st.session_state.get("diagnosis_inputs")
    if result and inputs:
        render_result(result, inputs)


if __name__ == "__main__":
    main()
