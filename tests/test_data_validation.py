import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for testing
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_raw_data, clean_data, COLUMN_MAPPING

def test_column_mapping():
    assert len(COLUMN_MAPPING) == 14
    assert "target" in COLUMN_MAPPING.values()
    assert "age" in COLUMN_MAPPING.values()

def test_load_and_clean_data(tmp_path):
    # Create dummy data resembling the raw file
    dummy_data = "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n" \
                 "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2\n" \
                 "67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1\n" \
                 "37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,0.0,3.0,0\n" \
                 "41.0,0.0,2.0,130.0,204.0,0.0,2.0,172.0,0.0,1.4,1.0,0.0,3.0,0\n" \
                 "56.0,1.0,3.0,130.0,256.0,1.0,2.0,142.0,1.0,0.6,2.0,1.0,6.0,3\n" \
                 "62.0,0.0,4.0,140.0,268.0,0.0,2.0,160.0,0.0,3.6,3.0,2.0,3.0,4\n" \
                 "57.0,0.0,4.0,120.0,354.0,0.0,0.0,163.0,1.0,0.6,1.0,?,3.0,0\n"
                 
    data_file = tmp_path / "processed.cleveland.data"
    data_file.write_text(dummy_data)
    
    # Test load
    df = load_raw_data(data_file)
    assert len(df) == 8
    assert list(df.columns) == list(COLUMN_MAPPING.values())
    
    # Check that '?' was converted to NaN
    assert pd.isna(df.loc[7, 'num_major_vessels'])
    
    # Test clean
    df_clean = clean_data(df)
    
    # Check target transformation
    assert df_clean.loc[0, 'target'] == 0
    assert df_clean.loc[1, 'target'] == 1 # Original was 2
    assert df_clean.loc[2, 'target'] == 1 # Original was 1
    assert df_clean.loc[3, 'target'] == 0
    assert df_clean.loc[5, 'target'] == 1 # Original was 3
    assert df_clean.loc[6, 'target'] == 1 # Original was 4
