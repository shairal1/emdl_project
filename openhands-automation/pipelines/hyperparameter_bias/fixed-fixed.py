# fixed-fixed.py
# Summary of fixes:
# - Removed data leakage by dropping 'dob', 'is_recid', 'two_year_recid' from features
# - Moved missing value imputation to after train/test split using training set statistics
# - Added stratify=y in train_test_split to preserve class distribution
# - Wrapped execution in main() with __name__ guard
# - Added numpy import for numeric imputation

import os
import sys
import pandas as pd
#FIXED
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    from utils import get_project_root
    project_root = get_project_root()

    raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
    raw_data = pd.read_csv(raw_data_file)

    #FIXED
    raw_data = raw_data[
        ['sex', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count',
         'days_b_screening_arrest', 'decile_score']
    ]

    X = raw_data.drop(columns=['score_text'])
    y = raw_data['score_text']

    #FIXED
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical_cols = X_train.select_dtypes(include=['object']).columns
    #FIXED
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns

    #FIXED
    for col in categorical_cols:
        X_train[col].fillna(X_train[col].mode()[0], inplace=True)
    #FIXED
        X_test[col].fillna(X_train[col].mode()[0], inplace=True)

    #FIXED
    for col in numeric_cols:
        X_train[col].fillna(X_train[col].mean(), inplace=True)
    #FIXED
        X_test[col].fillna(X_train[col].mean(), inplace=True)

    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    clf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    best_clf = RandomForestClassifier(**best_params, random_state=42)
    best_clf.fit(X_train, y_train)

    y_pred = best_clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

#FIXED
if __name__ == "__main__":
    main()
