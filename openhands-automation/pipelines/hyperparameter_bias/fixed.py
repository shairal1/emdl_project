# fixed.py

# Summary of fixes:
# 1. Use 'two_year_recid' as the target variable to predict actual recidivism instead of re-predicting COMPAS 'score_text'.
# 2. Remove leaky and redundant features: 'score_text', 'decile_score', 'is_recid', 'c_jail_in', 'c_jail_out', and 'dob'.
# 3. Filter records to keep only those with 'days_b_screening_arrest' in [-30, 30], as per COMPAS data guidelines.
# 4. Drop rows with missing target values instead of filling them, to avoid introducing noise.
# 5. Fill missing values only for feature columns: use mode for categorical and mean for numerical.
# 6. Perform stratified train/test split to preserve the distribution of the target classes.
# 7. Clearly align dummy-encoded train and test feature sets and print the classification report separately.

import pandas as pd
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Ensure utils can be imported from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

def main():
    # Locate data file
    project_root = get_project_root()
    raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
    raw_data = pd.read_csv(raw_data_file)

    # Filter to valid screening window
    raw_data = raw_data[
        (raw_data['days_b_screening_arrest'] >= -30) &
        (raw_data['days_b_screening_arrest'] <= 30)
    ]

    # Define features and target
    features = ['sex', 'age', 'c_charge_degree', 'race', 'priors_count', 'days_b_screening_arrest']
    target = 'two_year_recid'

    # Subset data and drop rows with missing target
    data = raw_data[features + [target]].dropna(subset=[target]).copy()

    # Fill missing feature values
    for col in features:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].mean())

    # Separate features and target
    X = data[features]
    y = data[target].astype(int)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # One-hot encode categorical features
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)

    # Align train and test feature sets
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
