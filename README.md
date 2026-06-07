# 🤖 DiagnostiX AI

An AI-powered Electronic Device Fault Prediction System built using Python, Streamlit, and Machine Learning.

## 🚀 Live Demo

[DiagnostiX AI Live App](YOUR_STREAMLIT_LINK_HERE)

---

## 📌 Features

- Predicts faulty electronic components
- Supports Mobile, Laptop, and Desktop devices
- Interactive Streamlit dashboard
- Machine Learning based prediction system
- Real-time fault detection interface
- User-friendly modern UI

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Pickle

---

## 📂 Project Structure

```text
DiagnostiX-AI/
│
├── app.py
├── diagnosis.py
├── train.py
├── model.pkl
├── tests/
├── requirements.txt
├── README.md
└── DiagnostiX_AI_600Plus_Dataset - DiagnostiX_600Rows.csv
```

## Run Locally

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

The saved model includes categorical preprocessing, input options, and the
classifier in one artifact. Run `python train.py` whenever the dataset changes.

## Prediction Inputs

- Device type and age
- Daily usage and usage type
- Month when the problem started
- Three observed symptoms

The app displays the most likely component, confidence, and two alternatives.

---

## 👨‍💻 Author

**Ashwin Dubey**

ECE Student | Chandigarh University


## 📧 Contact

Email:-cuniversity223@gmail.com
GitHub:- https://github.com/That-ASHWIN
LinkedIn:-https://www.linkedin.com/in/ashwin-dubey-b27657302/
