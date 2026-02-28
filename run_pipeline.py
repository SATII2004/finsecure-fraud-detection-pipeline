from src.pipeline.ingest import load_raw_data
from src.pipeline.process import process_data
from src.models.train import train_and_save_model
import os

def main():
    # 1. Ingest
    data_path = os.path.join('data', 'creditcard.csv')
    raw_df = load_raw_data(data_path)
    
    if raw_df is not None:
        # 2. Process (Silver/Gold Layers)
        X_train, X_test, y_train, y_test = process_data(raw_df)
        
        # 3. Train (AI Layer)
        train_and_save_model(X_train, X_test, y_train, y_test)
        
        print("\nPIPELINE EXECUTION COMPLETE.")

if __name__ == "__main__":
    main()