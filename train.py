import pandas as pd

# Load dataset
df = pd.read_csv("DiagnostiX_AI_600Plus_Dataset - DiagnostiX_600Rows.csv")

print("=== First 5 Rows ===")
print(df.head())

print("\n=== Dataset Shape ===")
print(df.shape)

print("\n=== Column Names ===")
print(df.columns.tolist())

print("\n=== Missing Values ===")
print(df.isnull().sum())
from sklearn.preprocessing import LabelEncoder

encoders = {}

categorical_columns = [
    "Device",
    "Usage_Type",
    "Symptom1",
    "Symptom2",
    "Symptom3",
    "Faulty_Component"
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print("\n=== Encoded Data ===")
print(df.head())
from sklearn.model_selection import train_test_split

# Features (input)
X = df.drop("Faulty_Component", axis=1)

# Target (output)
y = df["Faulty_Component"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Complete!")
from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy * 100, "%")
# Sample prediction

sample_data = [[
    1,   # Device (Mobile)
    24,  # Age_Months
    8,   # Daily_Usage_Hours
    22,  # Failure_After_Months
    1,   # Usage_Type
    24,  # Symptom1
    23,  # Symptom2
    1    # Symptom3
]]

prediction = model.predict(sample_data)

print("\nPrediction Code:", prediction[0])

fault_name = encoders["Faulty_Component"].inverse_transform(prediction)

print("Predicted Faulty Component:", fault_name[0])
import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoders["Faulty_Component"],
            open("fault_encoder.pkl", "wb"))
print("Model Saved Successfully!")
