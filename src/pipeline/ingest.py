import pandas as pd
import os

def load_raw_data(file_path):
    """
    Bronze Layer: Loading raw data from source.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    
    print("Ingesting data from source...")
    df = pd.read_csv(file_path)
    print(f"Successfully ingested {len(df)} records.")
    return df

if __name__ == "__main__":
    # Test the ingestion
    raw_df = load_raw_data('data/creditcard.csv')