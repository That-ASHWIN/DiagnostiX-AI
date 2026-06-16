# \U0001F6E0\uFE0F DiagnostiX AI - Electronic Device Fault Predictor

DiagnostiX AI is a machine-learning web app that suggests the **most likely
faulty hardware component** of an electronic device from a few simple inputs:
the device type, how it is used, when the problem started, and three symptoms
the user observes.

> \u26A0\uFE0F **It is a decision-support tool, not a final verdict.** The app points
> you toward the component most likely at fault so you know where to look first
> - always confirm with a physical hardware check before replacing anything.

## \U0001F517 Live Demo

Add your Streamlit link here after deploying:
`https://<your-app>.streamlit.app`

## \u2728 What it does

- Covers multiple device families (mobiles, laptops, desktops, tablets,
  smartwatches, routers, smart TVs, printers).
- Predicts the most likely faulty component and lists alternative suspects.
- Shows how strongly the reported symptoms match the predicted fault.
- Clean Streamlit interface that only offers symptom combinations the model
  has actually learned.

## \U0001F9E0 How it works

1. **Inputs** - device, age, daily usage, when the issue started, usage type,
   and three observed symptoms.
2. **Preprocessing** - categorical fields are one-hot encoded and numeric
   fields pass through (`ColumnTransformer`).
3. **Model** - a `RandomForestClassifier` (300 trees, class-balanced) wrapped
   in a scikit-learn `Pipeline`.
4. **Output** - the top predicted component, two alternatives, and a
   match-strength indicator.

The preprocessing, the model, and the list of valid inputs are bundled into a
single artifact so the app always stays consistent with how the model was
trained.

## \U0001F4CA Dataset & honest evaluation

The model is trained on a structured dataset of device-usage and repair
records that maps symptom patterns to faulty components.

**Being transparent:** this dataset is largely rule-based / pattern-driven, so
symptom combinations map very cleanly to faults. That means the measured
accuracy is **very high but partly optimistic** - on genuinely new, messy,
real-world symptom descriptions the model will be less certain. I would rather
state this openly than claim a flawless system.

Evaluation performed in `train.py`:

- Stratified train/test split
- 5-fold cross-validation
- Per-class precision / recall / F1 (`classification_report`)
- Confusion matrix image (`confusion_matrix.png`)

## \U0001F680 Run locally

```bash
pip install -r requirements.txt
python train.py        # optional: trains, prints full metrics, saves model.pkl
streamlit run app.py   # the app also self-trains on first run if no model exists
```

## \u2601\uFE0F Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub, including the dataset CSV.
2. Go to <https://share.streamlit.io> -> **New app** -> select this repo,
   branch `main`, main file `app.py`.
3. (Optional) In *Advanced settings* choose Python 3.12.
4. Deploy. On first load the app trains the model automatically and caches it.

## \U0001F52D Future scope

- Train on **real repair-shop data** instead of rule-based records.
- Handle **free-text / unseen symptoms** with fuzzy matching or text
  embeddings.
- **Confidence calibration** and a "not sure - needs inspection" fallback.
- A **feedback loop** so confirmed repairs improve the model over time.
- Per-component **repair cost & time estimates** (already in the dataset).

## \U0001F4C1 Project structure

```text
DiagnostiX-AI/
\u251C\u2500\u2500 app.py            # Streamlit UI
\u251C\u2500\u2500 diagnosis.py      # load/serve model + prediction helpers
\u251C\u2500\u2500 train.py          # training, evaluation, metrics
\u251C\u2500\u2500 tests/            # unit tests
\u251C\u2500\u2500 requirements.txt
\u2514\u2500\u2500 *.csv             # training dataset
```

## \U0001F468\u200D\U0001F4BB Author

**Ashwin Dubey** - ECE Student, Chandigarh University

- GitHub: <https://github.com/That-ASHWIN>
- LinkedIn: <https://www.linkedin.com/in/ashwin-dubey-b27657302/>
- Email: cuniversity223@gmail.com
