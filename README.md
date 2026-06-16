# 🛠️ DiagnostiX AI — Electronic Device Fault Predictor

Ever had a device just... stop working, and you had no clue *which part* actually gave up? That's the itch I built DiagnostiX AI to scratch.

It's a small machine-learning web app: you tell it the device, how old it is, how you use it, and the symptoms you're seeing — and it points you to the hardware component most likely at fault, so you know where to start looking.

> ⚠️ **It's a guide, not a guarantee.** DiagnostiX shows you the most likely culprit. Always confirm with a real hardware check before replacing anything.

## 🔗 Live Demo

> 🚧 **Not deployed yet.** DiagnostiX runs on Streamlit Community Cloud. Once you deploy it (see [Deploy](#️-deploy-on-streamlit-community-cloud) below), your live link goes right here — it'll look like `https://diagnostix-ai.streamlit.app`.

## ✨ What it can do

- Works across phones, laptops, desktops, tablets, smartwatches, routers, smart TVs and printers
- Predicts the most likely faulty component, plus a couple of backup suspects
- Tells you how strongly your symptoms match the prediction
- Only offers symptom combinations the model actually understands — so you never get nonsense inputs

## 🧠 How it works (the short version)

1. You enter the device, its age, daily usage, when the trouble started, and three symptoms.
2. Behind the scenes, the text fields get one-hot encoded and the numbers pass straight through (`ColumnTransformer`).
3. A **Random Forest** (300 trees, class-balanced) makes the call, all wrapped in one tidy scikit-learn `Pipeline`.
4. You get the top suspect, two alternatives, and a match-strength bar.

The model, the preprocessing, and the list of valid inputs all travel together in a single artifact — so the app and the model never drift out of sync.

## 📊 About the data

The dataset behind DiagnostiX is something I gathered from the ground up, rather than downloading off the shelf:

- **A Google Form survey** where everyday users described the problems they'd run into with their devices and what the fault eventually turned out to be.
- **Inputs from nearby hardware shops, service centres and repair technicians**, who shared the symptom-to-fault patterns they deal with every day.
- **Online problem reports** — common device issues that people post and discuss across the web, distilled into the same symptom-to-fault structure.

Combining real user reports, online complaints, and hands-on repair experience is what makes the predictions feel grounded in how devices actually fail, instead of being made-up numbers.

A small honesty note: because the collected data is fairly clean and the symptom patterns map quite consistently onto faults, the model scores very high on paper. Real-world symptoms are messier and more ambiguous, so treat the prediction as a strong starting hint — not a final verdict.

To keep the evaluation honest, `train.py` runs:

- a stratified train/test split
- 5-fold cross-validation
- per-class precision / recall / F1
- a confusion matrix you can actually look at

## 🚀 Run it locally

```bash
pip install -r requirements.txt
python train.py        # trains + prints the full metrics (optional)
streamlit run app.py   # the app trains itself on first run if no model exists
```

## ☁️ Deploy on Streamlit Community Cloud

1. Make sure the dataset CSV is in the repo.
2. Go to <https://share.streamlit.io> → **New app**.
3. Pick this repo, branch `main`, main file `app.py`.
4. (Optional) Choose Python 3.12 under *Advanced settings*.
5. Hit deploy — the app trains the model on first load and caches it.

## 🔭 Where I'd take it next

- Keep growing the dataset with **more survey responses and repair-shop records**
- Understand **free-text symptoms** using fuzzy matching / embeddings
- Add **confidence calibration** and an honest "I'm not sure — get it inspected" answer
- A **feedback loop** so every confirmed repair makes the model a little smarter
- Surface **repair cost & time** estimates alongside the predicted fault

## 📁 Project layout

```text
DiagnostiX-AI/
├── app.py          # the Streamlit interface
├── diagnosis.py    # loads/serves the model + prediction logic
├── train.py        # training, evaluation, metrics
├── tests/          # unit tests
├── requirements.txt
└── *.csv           # the training data
```

## 👨‍💻 About me

**Ashwin Dubey** — ECE student at Chandigarh University, learning by building.

- GitHub: <https://github.com/That-ASHWIN>
- LinkedIn: <https://www.linkedin.com/in/ashwin-dubey-b27657302/>
- Email: cuniversity223@gmail.com
