import pandas as pd
import numpy as np

def load_and_preprocess_data(data_path, severity_path):
    """Loads dataset and replaces text symptoms with numerical severity weights."""
    
    # Load data
    df = pd.read_csv(data_path)
    severity_df = pd.read_csv(severity_path)
    
    # --- THE FIX: Aggressively clean the text ---
    # This fixes the Kaggle typos by replacing spaces with underscores
    def clean_text(text):
        if isinstance(text, str):
            return text.strip().replace(' ', '_')
        return text
        
    df = df.map(clean_text)
    severity_df['Symptom'] = severity_df['Symptom'].map(clean_text)
    
    # Create a mapping dictionary
    severity_dict = dict(zip(severity_df['Symptom'], severity_df['weight']))
    
    # Separate features (X) and target (y)
    X = df.drop('Disease', axis=1)
    y = df['Disease']
    
    # Replace symptom text with weights
    X = X.replace(severity_dict)
    
    # --- SAFETY NET ---
    # Force any leftover unmapped strings to turn into NaN, then fill all NaNs with 0
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return X, y