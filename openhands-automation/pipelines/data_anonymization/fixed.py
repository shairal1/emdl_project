"""
Summary of fixes:
- Added import for numpy to handle missing values.
- Stripped whitespace from string columns to ensure consistent category values.
- Replaced placeholder '?' with NaN and dropped rows with missing values.
- Used separate LabelEncoder instances for each categorical feature to avoid shared state.
- Encoded target variable 'salary' separately and preserved class names for reporting.
- Dropped columns safely with errors='ignore' if they are not present.
- Added file existence check for the input data file.
- Wrapped execution in a main() function and used a __main__ guard.
- Set n_jobs=-1 in RandomForestClassifier for faster training.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


def main():
    # Ensure project root is on path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    # Import utility for locating project root
    from utils import get_project_root

    # Locate and verify data file
    project_root = get_project_root()
    raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
    if not os.path.isfile(raw_data_file):
        raise FileNotFoundError(f"Data file not found: {raw_data_file}")

    # Load data
    data = pd.read_csv(raw_data_file)

    # Drop irrelevant columns if present
    data.drop(columns=['education', 'occupation'], errors='ignore', inplace=True)

    # Clean whitespace and handle missing placeholders
    obj_cols = data.select_dtypes(include=['object']).columns
    for col in obj_cols:
        data[col] = data[col].str.strip()
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)

    # Encode categorical features
    feature_cols = data.columns.drop('salary')
    for col in feature_cols:
        if data[col].dtype == 'object':
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])

    # Encode target separately
    target_encoder = LabelEncoder()
    data['salary'] = target_encoder.fit_transform(data['salary'])

    # Split into features and target
    X = data.drop(columns=['salary'])
    y = data['salary']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))


if __name__ == '__main__':
    main()