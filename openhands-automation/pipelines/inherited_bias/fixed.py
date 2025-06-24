"""
fixed.py

Summary of fixes:
- Added na_values parameter to pd.read_csv to parse '?' as NaN and dropped missing values.
- Encoded target variable y using LabelEncoder for compatibility with LogisticRegression.
- Separated numeric and categorical preprocessing: numeric features imputed (mean) and scaled; categorical features imputed (most frequent) and one-hot encoded with sparse output disabled to avoid sparse matrix issues.
- Wrapped script execution in main() function and protected with if __name__ == '__main__' guard to allow safe imports.
- Replaced manual sys.path manipulation with a get_root_dir fallback using pathlib when utils.get_project_root is unavailable.
- Set random_state in LogisticRegression for reproducibility and specified solver='lbfgs'.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_root_dir():
    """Determine project root directory."""
    try:
        from utils import get_project_root
        return get_project_root()
    except ImportError:
        # Fallback: parent directory of this script
        return Path(__file__).resolve().parents[1]


def load_data(root_dir):
    """Load and preprocess raw data."""
    raw_data_file = Path(root_dir) / "datasets" / "adult_data" / "adult_data.csv"
    data = pd.read_csv(raw_data_file, na_values='?')
    # Drop rows with missing values
    data.dropna(inplace=True)
    return data


def build_pipeline(X):
    """Build a preprocessing and modeling pipeline."""
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

    # Numeric preprocessing: impute missing and scale
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical preprocessing: impute and one-hot encode (dense output)
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42))
    ])

    return model


def main():
    # Determine project root and load data
    project_root = get_root_dir()
    data = load_data(project_root)

    # Split features and target
    X = data.drop('salary', axis=1)
    y = data['salary']

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build and train pipeline
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print(classification_report(
        y_test, y_pred, zero_division=0, target_names=le.classes_
    ))


if __name__ == '__main__':
    main()