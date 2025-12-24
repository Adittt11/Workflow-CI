"""
Model Training with MLflow Autolog - Bank Churn
Author: Made Aditya Nugraha Arya Putra
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import argparse


def load_preprocessed_data(filepath):
    """Load preprocessed data"""
    print(f"Loading preprocessed data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded. Shape: {df.shape}")
    return df


def split_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nData split:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def main(data_path):
    """
    Main training pipeline

    Args:
        data_path: Path to preprocessed data
    """
    print("="*70)
    print("MODEL TRAINING - BANK CHURN CLASSIFICATION")
    print("="*70)

    # Load data
    df = load_preprocessed_data(data_path)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Enable MLflow autolog
    mlflow.sklearn.autolog()
    print("\nMLflow autolog enabled")

    print("\n" + "="*70)
    print("Training Random Forest Classifier")
    print("="*70)

    # Train model (MLflow run already creates a run context)
    print("\nTraining model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    # Evaluate on test set (autolog will log metrics automatically)
    test_score = model.score(X_test, y_test)
    print(f"\nTest Accuracy: {test_score:.4f}")

    print("\nModel trained successfully!")

    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with MLflow')
    parser.add_argument('--data', type=str, default='churn_preprocessed.csv',
                        help='Path to preprocessed data')

    args = parser.parse_args()

    try:
        main(args.data)
    except Exception as e:
        print(f"Error during training: {e}")
        raise
