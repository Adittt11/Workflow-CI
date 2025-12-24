"""
Automated Preprocessing Script for Bank Churn Dataset
Author: Made Aditya Nugraha Arya Putra
Purpose: Kriteria 2 - MSML Submission
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import argparse


def load_data(filepath):
    """
    Load dataset from CSV file

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame: Loaded dataset
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def check_data_quality(df):
    """
    Check data quality (missing values, duplicates)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: Checked DataFrame
    """
    print("\nChecking data quality...")

    # Check missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found:")
        print(missing[missing > 0])
    else:
        print("No missing values found")

    # Check duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")

    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows")

    return df


def drop_unnecessary_columns(df):
    """
    Drop columns not needed for modeling

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: DataFrame without unnecessary columns
    """
    print("\nDropping unnecessary columns...")
    columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']

    df = df.drop(columns=columns_to_drop)
    print(f"Dropped columns: {columns_to_drop}")
    print(f"Shape after dropping: {df.shape}")

    return df


def encode_categorical_features(df):
    """
    Encode categorical features using One-Hot Encoding

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: DataFrame with encoded categorical features
    """
    print("\nEncoding categorical features...")

    # One-Hot Encoding for Geography and Gender
    categorical_cols = ['Geography', 'Gender']

    print(f"Encoding columns: {categorical_cols}")
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    print(f"Shape after encoding: {df_encoded.shape}")

    return df_encoded


def scale_numerical_features(df):
    """
    Scale numerical features using StandardScaler

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: DataFrame with scaled numerical features
        StandardScaler: Fitted scaler
    """
    print("\nScaling numerical features...")

    numerical_features = [
        'CreditScore', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'EstimatedSalary'
    ]

    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    print(f"Scaled features: {numerical_features}")

    return df, scaler


def create_feature_engineering(df):
    """
    Create additional features (Age Group binning)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: DataFrame with new features
    """
    print("\nCreating feature engineering...")

    # Age Group binning
    # Note: Age is already scaled, so we need to use original bins mapped to scaled values
    # For simplicity, we'll create this as categorical based on quartiles
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=4,
        labels=['Young', 'Adult', 'Senior', 'Elder']
    )

    # Convert to numeric for modeling
    age_group_mapping = {'Young': 0, 'Adult': 1, 'Senior': 2, 'Elder': 3}
    df['Age_Group'] = df['Age_Group'].map(age_group_mapping)

    print("Created Age_Group feature")

    return df


def save_preprocessed_data(df, output_path):
    """
    Save preprocessed data to CSV file

    Args:
        df: Preprocessed DataFrame
        output_path: Path to save the CSV file
    """
    print(f"\nSaving preprocessed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Data saved successfully. Shape: {df.shape}")


def save_artifacts(scaler, output_dir):
    """
    Save preprocessing artifacts (scaler)

    Args:
        scaler: StandardScaler object
        output_dir: Directory to save artifacts
    """
    print("\nSaving preprocessing artifacts...")

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Save scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler.pkl")


def preprocess_pipeline(input_filepath, output_filepath, artifacts_dir):
    """
    Complete preprocessing pipeline

    Args:
        input_filepath: Path to input CSV file
        output_filepath: Path to save preprocessed CSV file
        artifacts_dir: Directory to save preprocessing artifacts
    """
    print("="*70)
    print("AUTOMATED PREPROCESSING PIPELINE - BANK CHURN")
    print("="*70)

    # Step 1: Load data
    df = load_data(input_filepath)

    # Step 2: Check data quality
    df = check_data_quality(df)

    # Step 3: Drop unnecessary columns
    df = drop_unnecessary_columns(df)

    # Step 4: Encode categorical features
    df = encode_categorical_features(df)

    # Step 5: Scale numerical features
    df, scaler = scale_numerical_features(df)

    # Step 6: Feature engineering
    df = create_feature_engineering(df)

    # Step 7: Save preprocessed data
    save_preprocessed_data(df, output_filepath)

    # Step 8: Save artifacts
    save_artifacts(scaler, artifacts_dir)

    print("\n" + "="*70)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Preprocessed data: {output_filepath}")
    print(f"Artifacts directory: {artifacts_dir}")
    print(f"Final shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess Bank Churn dataset')
    parser.add_argument('--input', type=str, default='churn.csv',
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, default='churn_preprocessed.csv',
                        help='Output CSV file path')
    parser.add_argument('--artifacts-dir', type=str, default='preprocessing/artifacts',
                        help='Directory to save preprocessing artifacts')

    args = parser.parse_args()

    # Run preprocessing pipeline
    preprocess_pipeline(
        input_filepath=args.input,
        output_filepath=args.output,
        artifacts_dir=args.artifacts_dir
    )
