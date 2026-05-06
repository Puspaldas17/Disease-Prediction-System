import joblib
import numpy as np
import pandas as pd

def predict_disease(symptoms_list, model_path='../models/random_forest_model.pkl'):
    """Loads the saved model and predicts the disease based on symptom weights."""
    
    # Load the trained model
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        return "Error: Model not found. Please run train.py first."

    # The Kaggle dataset uses exactly 17 symptom columns
    # We pad the user's input with zeros if they enter fewer than 17 symptoms
    features = np.zeros(17)
    
    for i, weight in enumerate(symptoms_list):
        if i < 17:
            features[i] = weight
            
    # Reshape for a single prediction (1 sample, 17 features)
    features_reshaped = features.reshape(1, -1)
    
    # Predict
    prediction = model.predict(features_reshaped)
    return prediction[0]

if __name__ == "__main__":
    # Example test: 
    # Let's say a patient has Itching (weight 1) and Skin Rash (weight 3)
    sample_symptoms = [1, 3] 
    
    result = predict_disease(sample_symptoms)
    print(f"The predicted disease is: {result}")