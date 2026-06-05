import streamlit as st
import pickle
st.set_page_config(
    page_title="DiagnostiX AI",
    page_icon="🛠️",
    layout="wide"
)
st.markdown("""
# 🤖 DiagnostiX AI

### Smart Electronic Device Fault Prediction System

Predict faulty components in Mobile, Laptop and Desktop devices using Machine Learning.

---
""")

with st.sidebar:
    st.header("📊 Device Info")
    st.write("DiagnostiX AI v1.0")
    st.write("Built with Python, Streamlit & Machine Learning")

st.subheader("Electronic Device Fault Prediction")

# Inputs
device = st.selectbox(
    "Select Device",
    ["Mobile", "Laptop", "Desktop"]
)

age = st.number_input("Age (Months)", min_value=1)
usage = st.number_input("Daily Usage Hours", min_value=1)

symptom1 = st.text_input("Symptom 1")
symptom2 = st.text_input("Symptom 2")
symptom3 = st.text_input("Symptom 3")

# Load model
model = pickle.load(open("model.pkl", "rb"))

if st.button("Predict"):

    # Temporary values
    sample_data = [[
        1,      # Device
        age,
        usage,
        22,     # Failure_After_Months
        1,      # Usage_Type
        24,     # Symptom1
        23,     # Symptom2
        1       # Symptom3
    ]]

    # Prediction
    prediction = model.predict(sample_data)

    st.success(f"Predicted Fault Code: {prediction[0]}")

    # Load encoder
    try:
        fault_encoder = pickle.load(open("fault_encoder.pkl", "rb"))

        fault_name = fault_encoder.inverse_transform(prediction)

        st.success(f"Faulty Component: {fault_name[0]}")

    except Exception as e:
        st.error(f"Encoder Error: {e}")