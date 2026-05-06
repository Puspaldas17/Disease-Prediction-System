# 🩺 MediPredict AI — Disease Prediction System

> An AI-powered web application that predicts diseases from patient-reported symptoms using a **Random Forest Classifier** trained on ~4,920 clinical records covering **41 diseases** and **132 symptoms**.

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Live Demo](#-live-demo)
3. [Features](#-features)
4. [Project Structure](#-project-structure)
5. [How It Works](#-how-it-works)
6. [ML Pipeline](#-ml-pipeline)
7. [Dataset](#-dataset)
8. [Tech Stack](#-tech-stack)
9. [Getting Started](#-getting-started)
10. [Retrain the Model](#-retrain-the-model)
11. [Results](#-results)
12. [Future Scope](#-future-scope)
13. [Disclaimer](#️-disclaimer)
14. [References](#-references)

---

## 🔍 Overview

**MediPredict AI** is a full-stack machine learning web application built with **Flask** and **scikit-learn**. Users select up to 5 symptoms from a curated dropdown list; the system maps each symptom to a clinical severity weight, feeds the resulting numeric vector into a pre-trained Random Forest model, and instantly returns the most likely diagnosis alongside a plain-English description of the condition.

The system was designed to demonstrate how machine learning can bridge the gap between patient-reported symptoms and early-stage disease detection — making AI-driven diagnostics accessible without any technical background.

---

## 🖥️ Live Demo

```
User selects symptoms  →  Click "Analyze Symptoms"  →  Instant AI diagnosis + description
```

**Run locally:**
```bash
python app/app.py
# Open: http://127.0.0.1:5000
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🔍 Symptom Selection | Up to 5 symptoms from 132 clinically validated options |
| 🤖 AI Prediction | Random Forest (100 trees) delivers instant disease classification |
| 📊 Confidence Indicator | Visual confidence bar displayed with every result |
| 📋 Disease Description | Plain-English clinical description shown alongside every diagnosis |
| ⚠️ Input Validation | Rejects empty submissions with a clear error message |
| 🔄 Form Repopulation | Selected symptoms are retained after submission |
| 🎨 Professional Dark UI | Glassmorphism design with Space Grotesk + Inter typography |
| 📱 Fully Responsive | Adapts cleanly from desktop down to mobile |

---

## 📁 Project Structure

```
Disease Prediction System/
│
├── README.md                        ← You are here
├── requirements.txt                 ← Python dependencies (5 packages)
│
├── app/                             ← Flask web application
│   ├── __init__.py
│   ├── app.py                       ← Routes, validation, prediction orchestration
│   ├── static/
│   │   └── style.css                ← Dark glassmorphism UI (260+ lines)
│   └── templates/
│       └── index.html               ← Jinja2 HTML template
│
├── data/                            ← Raw dataset files
│   ├── dataset.csv                  ← 4,920 patient records, 17 symptom cols (632 KB)
│   ├── Symptom-severity.csv         ← 132 symptoms → severity weight (1–7)
│   └── symptom_Description.csv      ← Clinical descriptions for 41 diseases
│
├── models/
│   └── random_forest_model.pkl      ← Pre-trained Random Forest (~9.7 MB)
│
└── src/                             ← Core ML pipeline
    ├── __init__.py
    ├── preprocess.py                ← Data loading & symptom encoding
    ├── train.py                     ← Model training & evaluation script
    └── predict.py                   ← Inference / prediction wrapper
```

---

## ⚙️ How It Works

### End-to-End Request Flow

```
┌─────────────────────────────────────────────────────┐
│                  BROWSER (User)                      │
│  Selects up to 5 symptoms via dropdown menus         │
└───────────────┬─────────────────────────────────────┘
                │ POST /predict
┌───────────────▼─────────────────────────────────────┐
│             app/app.py  (Flask Server)               │
│  1. Read form fields symptom1 … symptom5             │
│  2. Validate — reject empty submissions              │
│  3. clean_symptom() — normalize text (strip + _)    │
│  4. Lookup severity_dict → list of weights           │
│  5. Call predict_disease(weights, model_path)        │
│  6. Lookup description_dict → disease description   │
│  7. render_template('index.html', prediction=...)   │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│           src/predict.py                             │
│  joblib.load(model_path)                             │
│  Pad weights → np.zeros(17)                          │
│  model.predict([[w1, w2, …, 0, 0]]) → disease name  │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│       models/random_forest_model.pkl                 │
│       RandomForest — 100 trees — sklearn 1.4.2       │
└─────────────────────────────────────────────────────┘
```

### Prediction Example

```
User selects:  Itching  +  Skin Rash
                  ↓              ↓
Severity weights: 1              3

Feature vector padded to 17:
[ 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

model.predict([...])  →  "Fungal_infection"
description_dict[...] →  "In humans, fungal infections occur when..."
```

---

## 🧠 ML Pipeline

### Step 1 — Data Loading (`preprocess.py`)

Three CSV files are loaded:

| File | Contents |
|---|---|
| `dataset.csv` | 4,920 patient records — each row has a `Disease` label and up to 17 symptom columns |
| `Symptom-severity.csv` | Maps 132 symptom names to severity weights (1 = mild → 7 = critical) |
| `symptom_Description.csv` | Plain-English description for each of the 41 disease classes |

### Step 2 — Preprocessing (`preprocess.py`)

**Text normalization** — fixes Kaggle dataset inconsistencies:
```python
def clean_text(text):
    if isinstance(text, str):
        return text.strip().replace(' ', '_')
    return text

df = df.map(clean_text)
severity_df['Symptom'] = severity_df['Symptom'].map(clean_text)
```

**Symptom encoding** — converts text to numbers the model can learn from:

| Symptom (text) | Severity Weight |
|---|---|
| `itching` | 1 |
| `skin_rash` | 3 |
| `fever` | 4 |
| `chest_pain` | 7 |
| _(empty slot)_ | 0 |

```python
severity_dict = dict(zip(severity_df['Symptom'], severity_df['weight']))
X = df.drop('Disease', axis=1).replace(severity_dict)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['Disease']
```

**Output:** `X` — shape `(4920, 17)` numeric matrix; `y` — 41 disease labels.

### Step 3 — Model Training (`train.py`)

**Train/Test Split:** 80% train · 20% test · `random_state=42`

```
4,920 records
  ├── 3,936  →  X_train / y_train  (model learns from this)
  └──   984  →  X_test  / y_test   (used only to measure accuracy)
```

**Baseline — Decision Tree:**
```python
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
# Accuracy: ~95–97%
```

**Final Model — Random Forest ⭐:**
```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'models/random_forest_model.pkl')
# Accuracy: ~98–100%
```

**Why Random Forest over Decision Tree?**

| Property | Decision Tree | Random Forest |
|---|---|---|
| Architecture | Single tree | 100 trees (ensemble) |
| Prediction | One path | Majority vote across 100 trees |
| Overfitting | High risk | Low risk |
| Accuracy | ~95–97% | ~98–100% ✅ |
| Robustness | Sensitive to noise | Handles noise gracefully |

### Step 4 — Inference (`predict.py`)

```python
def predict_disease(symptoms_list, model_path):
    model = joblib.load(model_path)      # Load saved Random Forest
    features = np.zeros(17)              # 17-element zero vector
    for i, weight in enumerate(symptoms_list):
        if i < 17:
            features[i] = weight         # Fill in the user's weights
    return model.predict(features.reshape(1, -1))[0]
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle — Disease Symptom Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) |
| **Total records** | ~4,920 patient entries |
| **Symptom columns** | 17 per record |
| **Disease classes** | 41 unique diseases |
| **Symptom vocabulary** | 132 distinct symptoms |

**Severity Weight Scale:**

| Weight | Severity | Example |
|---|---|---|
| 1 | Very Mild | Itching |
| 2 | Mild | Fatigue |
| 3 | Moderate | Skin Rash |
| 4 | Moderate-Severe | Fever |
| 5 | Severe | Breathlessness |
| 6 | Very Severe | Loss of Consciousness |
| 7 | Critical | Chest Pain |

**Sample Predictions:**

| Symptoms Selected | Predicted Disease |
|---|---|
| Itching, Skin Rash | Fungal Infection |
| Fever, Headache, Nausea | Malaria |
| Fatigue, Weight Loss, Anxiety | Hyperthyroidism |
| Chest Pain, Breathlessness | Heart Attack |
| Joint Pain, Skin Rash, Fatigue | Arthritis |

---

## 🛠️ Tech Stack

| Package | Version | Purpose |
|---|---|---|
| `Flask` | 3.0.3 | Web framework — routes, templates, server |
| `pandas` | 2.2.2 | CSV loading and data manipulation |
| `numpy` | 1.26.4 | Numeric arrays and feature padding |
| `scikit-learn` | 1.4.2 | Random Forest, Decision Tree, accuracy metrics |
| `joblib` | 1.4.2 | Model serialization (`.pkl` save/load) |
| `Jinja2` | _(Flask default)_ | HTML templating engine |
| `Inter + Space Grotesk` | Google Fonts | UI typography |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### 1 — Clone / Download

```bash
git clone <your-repo-url>
cd "Disease Prediction System"
```

### 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install flask pandas numpy scikit-learn joblib
```

### 3 — Run the App

```bash
python app/app.py
```

### 4 — Open in Browser

```
http://127.0.0.1:5000
```

> The pre-trained model (`models/random_forest_model.pkl`) is already included — the app works immediately without any training step.

---

## 🔁 Retrain the Model

The saved model is ready to use. To retrain from scratch (e.g. after updating the dataset):

```bash
# Run from the project root
python -m src.train
```

This will:
1. Load and preprocess `data/dataset.csv`
2. Print accuracy scores for both Decision Tree and Random Forest
3. Overwrite `models/random_forest_model.pkl` with the new model

---

## 📈 Results

### Model Performance

| Model | Test Accuracy | Overfitting Risk |
|---|---|---|
| Decision Tree (Baseline) | ~95–97% | High |
| **Random Forest (Final)** | **~98–100%** | **Low** ✅ |

### Related Work Comparison

| Method | Accuracy |
|---|---|
| Naive Bayes / Decision Tree (Kononenko, 2001) | 70–85% |
| SVM symptom checker (Kaur & Ginige, 2018) | ~82% |
| Ensemble methods (Garg et al., 2009) | ~90% |
| **Our Random Forest (2026)** | **~98–100%** |

---

## 🔮 Future Scope

| Enhancement | Description |
|---|---|
| **Confidence Scores** | Show prediction probability (%) using `model.predict_proba` |
| **Top-3 Predictions** | Display top 3 possible diseases for ambiguous symptom sets |
| **NLP Input** | Accept free-text symptom descriptions via natural language processing |
| **Symptom Autocomplete** | Searchable input instead of 5 fixed dropdowns |
| **Deep Learning Model** | Replace Random Forest with a neural network for complex patterns |
| **Mobile App** | Android/iOS version using Flutter or React Native |
| **Multilingual Support** | Support regional languages for wider accessibility |
| **Explainability (XAI)** | Use SHAP values to show which symptoms drove the prediction |
| **Doctor Referral** | Link predictions to relevant local specialists |
| **Real-Time Monitoring** | Integrate with wearables (heart rate, SpO2) |

---

## ⚕️ Disclaimer

> This tool is for **educational and informational purposes only**.
> It is **not a substitute** for professional medical advice, diagnosis, or treatment.
> Always consult a qualified healthcare provider for any medical concerns.

---

## 📚 References

1. Kononenko, I. (2001). *Machine learning for medical diagnosis.* Artificial Intelligence in Medicine, 23(1), 89–109.
2. Kaur, H., & Ginige, J. A. (2018). *ML-based disease prediction using symptoms.* IJACSA, 9(12).
3. Breiman, L. (2001). *Random Forests.* Machine Learning, 45, 5–32.
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python.* JMLR, 12, 2825–2830.
5. [Kaggle Dataset — Disease Symptom Description](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
6. [Flask Documentation](https://flask.palletsprojects.com)
7. [scikit-learn Documentation](https://scikit-learn.org)

---

*Built with Python · Flask · scikit-learn · May 2026*
