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
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Devices Supported", "3")

with col2:
    st.metric("ML Model", "Active")

with col3:
    st.metric("Status", "Online")

with st.sidebar:
    st.header("📊 Device Info")
    st.write("DiagnostiX AI v1.0")
    st.write("Built with Python, Streamlit & Machine Learning")

st.subheader("Electronic Device Fault Prediction")
st.markdown("""
<style>
div.stButton > button {
    width: 100%;
    height: 3em;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Inputs
device = st.selectbox(
    "📱💻🖥 Select Device",
    ["Mobile","Laptop","Desktop"]
)

age = st.number_input("Age (Months)", min_value=1)
usage = st.number_input("Daily Usage Hours", min_value=1)

symptom1 = st.text_input("Symptom 1")
symptom2 = st.text_input("Symptom 2")
symptom3 = st.text_input("Symptom 3")

# Load model
model = pickle.load(open("model.pkl", "rb"))

col1, col2, col3 = st.columns([1,2,1])

with col2:
    predict_btn = st.button("🚀 Predict Fault")
st.markdown("""
<style>
.stButton > button {
    width: 100%;
    height: 55px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)
if predict_btn:

    # Temporary values
    sample_data = [[
        1,      # Device
        age,
        usage,
        22,     # Failure_After_Months
        1,      # Usage_Type
        24,     # Symptom1
        23,     # Symptom23
        1       # Symptom3
    ]]

        # Prediction
        prediction = model.predict(sample_data)

        st.markdown("## 📊 Prediction Results")
        st.info(f"📟 Predicted Fault Code: {prediction[0]}")

        try:
            fault_encoder = pickle.load(open("fault_encoder.pkl", "rb"))

            fault_name = fault_encoder.inverse_transform(prediction)

            st.success(f"🛠️ Faulty Component: {fault_name[0]}")
            st.balloons()

        except Exception as e:
            st.error(f"Encoder Error: {e}")
st.markdown("---")
st.markdown("Made by Ashwin Dubey | DiagnostiX AI 🚀")