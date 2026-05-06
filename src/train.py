import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess_data

# --- BULLETPROOF PATHS ---
# This automatically finds your main project folder no matter where you run the script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def train_models():
    # 1. Load the cleaned data safely
    print("Loading and preprocessing data...")
    data_path = os.path.join(BASE_DIR, 'data', 'dataset.csv')
    severity_path = os.path.join(BASE_DIR, 'data', 'Symptom-severity.csv')
    
    X, y = load_and_preprocess_data(data_path, severity_path)
    
    # 2. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train Baseline (Decision Tree)
    print("Training Decision Tree Baseline...")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_preds) * 100:.2f}%")
    
    # 4. Train Upgrade (Random Forest)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds) * 100:.2f}%")
    
    # 5. Save the Random Forest model safely into the models folder
    model_path = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
    joblib.dump(rf_model, model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    train_models()