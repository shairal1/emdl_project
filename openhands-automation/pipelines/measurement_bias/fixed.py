# Summary of fixes:
# - Removed dependency on external utils module and sys.path manipulation; use pathlib to locate project root.
# - Added error handling for missing data file and missing target column.
# - Added stratification in train_test_split to preserve class distribution.
# - Added feature scaling using StandardScaler and pipeline to improve model performance.
# - Specified solver and random_state in LogisticRegression for reproducibility.
# - Set zero_division=0 in classification_report to avoid division-by-zero warnings.

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main():
    # Determine project root as the directory containing this script
    project_root = Path(__file__).resolve().parent

    # Construct the path to the dataset
    data_path = project_root / "datasets" / "diabetes_indicator" / "binary_health_indicators.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load the data
    data = pd.read_csv(data_path)
    target_col = "Diabetes_binary"
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Split the data with stratification to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define a pipeline with scaling and logistic regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(solver="liblinear", max_iter=1000, random_state=42))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

if __name__ == "__main__":
    main()
