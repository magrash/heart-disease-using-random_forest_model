"""
Disease Prediction Model Training Script
=========================================
This script trains a Random Forest classifier on disease symptom data.

Usage:
    python train_model.py

Output:
    - random_forest_model.pkl: Trained model file
    - Confusion matrix visualizations
    - Accuracy metrics for training, validation, and test sets
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os


def load_data(train_path: str, test_path: str) -> tuple:
    """
    Load training and testing datasets.
    
    Args:
        train_path: Path to training CSV file
        test_path: Path to testing CSV file
    
    Returns:
        Tuple of (train_df, test_df)
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Drop unnecessary column if exists
    if "Unnamed: 133" in train.columns:
        train = train.drop("Unnamed: 133", axis=1)
    
    return train, test


def check_data_quality(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Check and report data quality issues."""
    print("=" * 50)
    print("DATA QUALITY CHECK")
    print("=" * 50)
    
    train_missing = train.isna().sum().sum()
    test_missing = test.isna().sum().sum()
    
    print(f"Training data shape: {train.shape}")
    print(f"Testing data shape: {test.shape}")
    print(f"Training missing values: {train_missing}")
    print(f"Testing missing values: {test_missing}")
    print()


def prepare_data(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Prepare data for model training.
    
    Returns:
        Tuple of (X_train, X_valid, y_train, y_valid, X_test, y_test)
    """
    # Separate features and target
    X = train.drop(["prognosis"], axis=1)
    y = train["prognosis"]
    
    X_test = test.drop(["prognosis"], axis=1)
    y_test = test["prognosis"]
    
    # Split training data into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_valid, y_train, y_valid, X_test, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained RandomForestClassifier model
    """
    print("=" * 50)
    print("TRAINING MODEL")
    print("=" * 50)
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    model.fit(X_train, y_train)
    print("Model training complete!")
    print()
    
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """Evaluate model performance on all datasets."""
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    # Predictions
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)
    
    # Accuracy scores
    print(f"Training Accuracy:   {accuracy_score(y_train, train_pred):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_valid, valid_pred):.4f}")
    print(f"Testing Accuracy:    {accuracy_score(y_test, test_pred):.4f}")
    print()
    
    # Plot confusion matrices
    plot_confusion_matrix(y_valid, valid_pred, model.classes_, "Validation")
    plot_confusion_matrix(y_test, test_pred, model.classes_, "Test")


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    classes: np.ndarray,
    title: str
) -> None:
    """Plot and display a confusion matrix."""
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {title} Set')
    plt.tight_layout()
    plt.show()


def save_model(model: RandomForestClassifier, filename: str) -> None:
    """Save the trained model to a file."""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")
    print(f"Model file size: {os.path.getsize(filename) / (1024*1024):.2f} MB")


def main():
    """Main function to orchestrate model training."""
    # File paths (relative to project directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, 'Training.csv')
    test_path = os.path.join(script_dir, 'Testing.csv')
    model_path = os.path.join(script_dir, 'random_forest_model.pkl')
    
    # Load data
    train_df, test_df = load_data(train_path, test_path)
    
    # Check data quality
    check_data_quality(train_df, test_df)
    
    # Prepare data
    X_train, X_valid, y_train, y_valid, X_test, y_test = prepare_data(train_df, test_df)
    
    print(f"Training set:   {X_train.shape[0]} samples")
    print(f"Validation set: {X_valid.shape[0]} samples")
    print(f"Test set:       {X_test.shape[0]} samples")
    print()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
    
    # Save model
    save_model(model, model_path)
    
    print()
    print("=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
