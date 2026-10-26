import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_evaluate():
    data_path = 'data/student_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError("Dataset missing. Run src/generate_data.py first.")
        
    df = pd.read_csv(data_path)
    
    X = df[['cgpa', 'dsa_problems_solved', 'projects_count', 'internships', 'soft_skills_score']]
    y = df['placed']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model Training
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/placement_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("💾 Model and Scaler saved successfully in models/")

if __name__ == '__main__':
    train_and_evaluate()