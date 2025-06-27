"""
Summary of fixes:
- Added stratified sampling in train_test_split to maintain class distribution.
- Handled missing values by dropping rows with any missing data.
- Wrapped script in a main() function and added the standard entry point guard.
- Replaced sys.path manipulation with robust project root detection via pathlib.
- Added exception handling for file I/O operations.
- Configured RandomForestClassifier to utilize all CPU cores with n_jobs=-1.
- Switched to using Python's logging module for status reporting.
"""

import sys
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Attempt to import helper; fallback if utils is not on path
try:
    from utils import get_project_root
except ImportError:
    def get_project_root():
        # Fallback: assume this script is two levels below project root
        return Path(__file__).resolve().parents[1]


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    project_root = Path(get_project_root())
    data_path = project_root / "datasets" / "diabetes_indicator" / "binary_health_indicators.csv"
    if not data_path.is_file():
        logging.error(f"Data file not found: {data_path}")
        sys.exit(1)

    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        logging.error(f"Could not read CSV file: {e}")
        sys.exit(1)

    # Drop rows with missing values
    if data.isnull().values.any():
        logging.warning("Missing values detected; dropping incomplete rows.")
        data = data.dropna()

    X = data.drop(columns=['Diabetes_binary'])
    y = data['Diabetes_binary']

    # Stratified split to preserve target distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info("Classification Report:\n%s", classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
