#!/usr/bin/env python3
"""
Script to prepare input data for modelV3 API testing.
Extracts only the 78 features used by modelV3 and filters to DoS classes only.
Creates input_data/ folder with CSV files ready for API testing.
"""

import os
import glob
import pandas as pd
import numpy as np

# Configuration
DATA_DIR = os.path.join(os.getcwd(), 'datasets')
OUTPUT_DIR = os.path.join(os.getcwd(), 'input_data')
LABEL_COL = 'Label'
DOS_CLASSES = ['BENIGN', 'DDoS', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris']
MIN_SAMPLES_PER_CLASS = 1000

# Load feature order from scaler stats (must match API expectations)
SCALER_PATH = os.path.join(os.getcwd(), 'artifacts', 'scaler_stats.npz')

def load_feature_order():
    """Load the exact feature order from scaler stats"""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler stats not found at {SCALER_PATH}. Run modelV3 first!")
    
    scaler_data = np.load(SCALER_PATH, allow_pickle=True)
    feature_columns = list(scaler_data['columns'])
    print(f"Loaded {len(feature_columns)} features from scaler stats")
    return feature_columns

def prepare_input_data():
    """Process datasets and create input_data folder"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load feature order
    required_features = load_feature_order()
    
    # Find all CSV files
    csv_paths = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    
    print(f"\nFound {len(csv_paths)} CSV files")
    
    # Process each CSV file
    all_processed = []
    
    for csv_path in csv_paths:
        filename = os.path.basename(csv_path)
        print(f"\nProcessing: {filename}")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            print(f"  Original shape: {df.shape}")
            
            # Normalize column names (strip whitespace)
            df.columns = [str(c).strip() for c in df.columns]
            
            # Filter to DoS classes only
            if LABEL_COL not in df.columns:
                print(f"  WARNING: No '{LABEL_COL}' column, skipping")
                continue
            
            mask = df[LABEL_COL].isin(DOS_CLASSES)
            df_filtered = df[mask].copy()
            print(f"  After DoS filter: {df_filtered.shape}")
            
            if len(df_filtered) == 0:
                print(f"  No DoS classes found, skipping")
                continue
            
            # Check which required features are available
            missing_features = []
            available_features = []
            
            for feat in required_features:
                if feat in df_filtered.columns:
                    available_features.append(feat)
                else:
                    missing_features.append(feat)
            
            if missing_features:
                print(f"  WARNING: Missing {len(missing_features)} features:")
                print(f"    {missing_features[:5]}...")
                # Fill missing features with 0
                for feat in missing_features:
                    df_filtered[feat] = 0.0
                    available_features.append(feat)
            
            # Select features in the exact order required by the API
            df_output = df_filtered[required_features + [LABEL_COL]].copy()
            
            # Clean data (replace inf/nan)
            for col in required_features:
                df_output[col] = pd.to_numeric(df_output[col], errors='coerce')
                df_output[col] = df_output[col].replace([np.inf, -np.inf], np.nan)
                df_output[col] = df_output[col].fillna(0.0)
            
            # Save to output directory
            output_path = os.path.join(OUTPUT_DIR, filename)
            df_output.to_csv(output_path, index=False)
            print(f"  Saved: {output_path} ({df_output.shape})")
            
            # Count classes
            class_counts = df_output[LABEL_COL].value_counts()
            print(f"  Class distribution:")
            for cls, count in class_counts.items():
                print(f"    {cls}: {count}")
            
            all_processed.append(df_output)
            
        except Exception as e:
            print(f"  ERROR processing {filename}: {str(e)}")
            continue
    
    # Create a combined file
    if all_processed:
        print(f"\n{'='*60}")
        print("Creating combined file...")
        combined_df = pd.concat(all_processed, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DIR, 'combined_dos_data.csv')
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined file: {combined_path}")
        print(f"Total samples: {len(combined_df)}")
        print(f"Features: {len(required_features)}")
        print(f"Classes: {sorted(combined_df[LABEL_COL].unique())}")
        
        # Summary statistics
        print(f"\nClass distribution in combined file:")
        class_counts = combined_df[LABEL_COL].value_counts()
        for cls, count in class_counts.items():
            pct = 100 * count / len(combined_df)
            print(f"  {cls}: {count:,} ({pct:.2f}%)")
    
    print(f"\n{'='*60}")
    print(f"✓ Input data preparation complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files created: {len(all_processed)} individual + 1 combined")
    print(f"\nTo test the API, use files from: {OUTPUT_DIR}")
    print("The API expects CSV files with these 78 features (in order):")
    print(f"  {required_features[:5]}... (and 73 more)")

if __name__ == "__main__":
    prepare_input_data()

