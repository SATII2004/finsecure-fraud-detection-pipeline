import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def process_data(df):
    """
    Silver Layer: Cleaning and Normalization
    Gold Layer: Feature Engineering & Balancing
    """
    print("Starting Data Transformation...")
    
    # 1. Normalize 'Amount' and 'Time' (Scaling features to a similar range)
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # 2. Split Features and Target
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    
    # 3. SMOTE: Oversampling the fraud cases so the AI learns better
    print("Balancing dataset using SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    
    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )
    
    print("Data processed and balanced successfully!")
    return X_train, X_test, y_train, y_test