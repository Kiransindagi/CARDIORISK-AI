import requests
import os
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_FILE = RAW_DATA_DIR / "processed.cleveland.data"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

def download_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        print(f"Data already exists at {DATA_FILE}")
        return
        
    print(f"Downloading data from {URL}...")
    response = requests.get(URL)
    
    if response.status_code == 200:
        with open(DATA_FILE, "wb") as f:
            f.write(response.content)
        print(f"Data successfully downloaded and saved to {DATA_FILE}")
    else:
        print(f"Failed to download data. HTTP Status code: {response.status_code}")
        sys.exit(1)

if __name__ == "__main__":
    download_data()
