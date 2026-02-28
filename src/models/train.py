import xgboost as xgb
import joblib
import shap
import os
from sklearn.metrics import classification_report

def train_and_save_model(X_train, X_test, y_train, y_test):
    print("Initializing XGBoost Classifier...")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # --- NEW: AI EXPLAINABILITY SECTION ---
    print("Creating SHAP Explainer...")
    explainer = shap.TreeExplainer(model)
    
    # Save both the Model and the Explainer
    model_path = os.path.join('src', 'models', 'fraud_model.pkl')
    explainer_path = os.path.join('src', 'models', 'shap_explainer.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(explainer, explainer_path)
    
    print(f"Model and Explainer saved successfully!")
    return model