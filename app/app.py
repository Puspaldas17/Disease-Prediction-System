from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import sys
import os

# --- BULLETPROOF PATHS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.predict import predict_disease

app = Flask(__name__)

# Secret key required for session — reads from env var in production
app.secret_key = os.environ.get('SECRET_KEY', 'medipredict-dev-secret-key')

# ------------------------------------------------------------------ #
# Shared normalization helper — matches preprocess.py exactly         #
# ------------------------------------------------------------------ #
def clean_symptom(text):
    """Strip whitespace and replace spaces with underscores."""
    if isinstance(text, str):
        return text.strip().replace(' ', '_')
    return text

# Load & normalize symptom severity data (once at startup)
csv_path = os.path.join(BASE_DIR, 'data', 'Symptom-severity.csv')
severity_df = pd.read_csv(csv_path)
severity_df['Symptom'] = severity_df['Symptom'].apply(clean_symptom)
symptoms_list = severity_df['Symptom'].tolist()

# Load disease descriptions (once at startup)
desc_path = os.path.join(BASE_DIR, 'data', 'symptom_Description.csv')
desc_df = pd.read_csv(desc_path)
description_dict = {
    row['Disease'].strip().replace(' ', '_'): row['Description'].strip()
    for _, row in desc_df.iterrows()
}


@app.route('/')
def home():
    # Pop result from session — consumed once, gone on refresh
    prediction         = session.pop('prediction', None)
    disease_description = session.pop('disease_description', None)
    error_text         = session.pop('error_text', None)
    user_choices       = session.pop('user_choices', None)

    return render_template(
        'index.html',
        symptoms=symptoms_list,
        prediction=prediction,
        disease_description=disease_description,
        error_text=error_text,
        user_choices=user_choices
    )


@app.route('/predict', methods=['POST'])
def predict():
    # Read dropdown selections
    user_choices = {
        1: request.form.get('symptom1', ''),
        2: request.form.get('symptom2', ''),
        3: request.form.get('symptom3', ''),
        4: request.form.get('symptom4', ''),
        5: request.form.get('symptom5', '')
    }

    selected_symptoms = [val for val in user_choices.values() if val != '']

    # Validate — reject empty submissions
    if not selected_symptoms:
        session['error_text']   = 'Please select at least one symptom before analyzing.'
        session['user_choices'] = user_choices
        return redirect(url_for('home'))

    # Convert symptoms → severity weights
    severity_dict = dict(zip(severity_df['Symptom'], severity_df['weight']))
    symptom_weights = []
    for sym in selected_symptoms:
        normalized = clean_symptom(sym)
        if normalized in severity_dict:
            symptom_weights.append(severity_dict[normalized])

    if not symptom_weights:
        session['error_text']   = 'None of the selected symptoms could be matched. Please try different symptoms.'
        session['user_choices'] = user_choices
        return redirect(url_for('home'))

    # Run prediction
    model_file = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
    prediction = predict_disease(symptom_weights, model_path=model_file)

    # Lookup disease description
    normalized_prediction = str(prediction).strip().replace(' ', '_')
    disease_description = description_dict.get(
        normalized_prediction,
        'No description is available for this condition.'
    )

    # Store result in session, then redirect to GET /
    # → refreshing the result page will show a clean form (PRG pattern)
    session['prediction']          = str(prediction)
    session['disease_description'] = disease_description
    session['user_choices']        = user_choices
    return redirect(url_for('home'))


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug)