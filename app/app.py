from flask import Flask, render_template, request
import pandas as pd
import sys
import os

# --- BULLETPROOF PATHS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.predict import predict_disease

app = Flask(__name__)

# ------------------------------------------------------------------ #
# FIX 1: Shared normalization helper — matches preprocess.py exactly  #
# ------------------------------------------------------------------ #
def clean_symptom(text):
    """Strip whitespace and replace spaces with underscores."""
    if isinstance(text, str):
        return text.strip().replace(' ', '_')
    return text

# Load & normalize symptom severity data
csv_path = os.path.join(BASE_DIR, 'data', 'Symptom-severity.csv')
severity_df = pd.read_csv(csv_path)
severity_df['Symptom'] = severity_df['Symptom'].apply(clean_symptom)  # FIX 1 applied
symptoms_list = severity_df['Symptom'].tolist()

# ------------------------------------------------------------------ #
# FIX 2: Load disease descriptions (was loaded but never used before) #
# ------------------------------------------------------------------ #
desc_path = os.path.join(BASE_DIR, 'data', 'symptom_Description.csv')
desc_df = pd.read_csv(desc_path)
# Normalize disease keys: lowercase + underscores so they match model output
description_dict = {
    row['Disease'].strip().replace(' ', '_'): row['Description'].strip()
    for _, row in desc_df.iterrows()
}


@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms_list)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        # Save dropdown selections to repopulate form after submit
        user_choices = {
            1: request.form.get('symptom1', ''),
            2: request.form.get('symptom2', ''),
            3: request.form.get('symptom3', ''),
            4: request.form.get('symptom4', ''),
            5: request.form.get('symptom5', '')
        }

        selected_symptoms = [val for val in user_choices.values() if val != ""]

        # ---------------------------------------------------------- #
        # FIX 3: Input validation — reject zero-symptom submissions   #
        # ---------------------------------------------------------- #
        if not selected_symptoms:
            return render_template(
                'index.html',
                symptoms=symptoms_list,
                error_text="Please select at least one symptom before analyzing.",
                user_choices=user_choices
            )

        # Build severity lookup from already-normalized dataframe
        severity_dict = dict(zip(severity_df['Symptom'], severity_df['weight']))

        # Map selected symptoms to weights; skip any that don't match (with warning)
        symptom_weights = []
        unmatched = []
        for sym in selected_symptoms:
            normalized = clean_symptom(sym)  # FIX 1: normalize before lookup
            if normalized in severity_dict:
                symptom_weights.append(severity_dict[normalized])
            else:
                unmatched.append(sym)

        if not symptom_weights:
            return render_template(
                'index.html',
                symptoms=symptoms_list,
                error_text="None of the selected symptoms could be matched. Please try different symptoms.",
                user_choices=user_choices
            )

        model_file = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
        prediction = predict_disease(symptom_weights, model_path=model_file)

        # FIX 2: Retrieve disease description for the predicted disease
        # Normalize prediction (model may return spaces or underscores) before lookup
        normalized_prediction = str(prediction).strip().replace(' ', '_')
        disease_description = description_dict.get(
            normalized_prediction,
            "No description is available for this condition."
        )

        return render_template(
            'index.html',
            symptoms=symptoms_list,
            prediction=prediction,
            disease_description=disease_description,
            user_choices=user_choices
        )


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug)