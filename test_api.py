import requests
import json

# URL of your local Flask API
url = "http://127.0.0.1:5000/predict"

# Example transaction data (30 features similar to the Kaggle dataset)
# This simulates one row of data being sent to your "Data Engineer" pipeline
sample_data = {
    "features": [0.0, -1.35, 1.19, 1.37, -0.06, 0.46, 0.28, -0.01, 0.67, -0.18, 
                 -0.39, 0.59, -0.18, -1.13, 0.18, -0.38, 0.48, 0.54, -0.01, 0.40, 
                 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 149.62]
}

print("Sending transaction to FinSecure Engine...")
response = requests.post(url, json=sample_data)

print("\n--- Response from Server ---")
print(response.json())