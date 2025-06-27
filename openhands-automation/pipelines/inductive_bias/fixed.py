#!/usr/bin/env python3
"""
Summary of fixes:
- Encoded target variable 'salary' using LabelEncoder to convert string labels to numeric.
- Handled missing values ('?') by replacing with NaN and dropping rows.
- Updated OneHotEncoder to use sparse=False for backward compatibility.
- Wrapped execution in main() with __main__ guard.
- Set random_state for reproducibility.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils import get_project_root

def main():
    # Determine project root and load data
    project_root = get_project_root()
    raw_data_file = os.path.join(
        project_root, "datasets", "adult_data", "adult_data.csv"
    )
    data = pd.read_csv(raw_data_file)

    # Handle missing values represented by '?'
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)

    # Separate features and target
    X = data.drop('salary', axis=1)
    y = data['salary']

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split into train/test sets
    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Identify categorical and numeric columns
    cat_cols = X_train.select_dtypes(include=['object']).columns
    num_cols = X_train.select_dtypes(exclude=['object']).columns

    # One-hot encode categorical features
    ohe = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    X_train_cat = ohe.fit_transform(X_train[cat_cols])
    X_test_cat = ohe.transform(X_test[cat_cols])

    # Combine numeric and encoded categorical features
    X_train_final = pd.concat([
        X_train[num_cols].reset_index(drop=True),
        pd.DataFrame(
            X_train_cat,
            columns=ohe.get_feature_names_out(cat_cols)
        ).reset_index(drop=True)
    ], axis=1)
    X_test_final = pd.concat([
        X_test[num_cols].reset_index(drop=True),
        pd.DataFrame(
            X_test_cat,
            columns=ohe.get_feature_names_out(cat_cols)
        ).reset_index(drop=True)
    ], axis=1)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, C=1e-4, random_state=42)
    model.fit(X_train_final, y_train_enc)

    # Predict and evaluate
    y_pred_enc = model.predict(X_test_final)
    print(
        classification_report(
            y_test_enc,
            y_pred_enc,
            target_names=label_encoder.classes_
        )
    )

if __name__ == '__main__':
    main()
