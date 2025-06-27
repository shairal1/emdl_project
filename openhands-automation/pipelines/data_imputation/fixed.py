"""
Summary of fixes:
1. Removed unnecessary sys.path manipulation and rely on get_project_root for locating project root.
2. Dropped potential leakage features ('decile_score', 'score_factor') and identifiers to avoid data leakage.
3. Added stratify=y in train_test_split to preserve class distribution.
4. Replaced manual LabelEncoder for features with OneHotEncoder in a ColumnTransformer.
5. Applied median imputation for numeric features and most frequent for categorical features.
6. Integrated preprocessing and classification into a sklearn Pipeline for consistency.
7. Set RandomForestClassifier n_jobs=-1 for performance and random_state=42 for reproducibility.
8. Removed manual LabelEncoder for target; classifier now handles string labels directly.
"""
import os
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils import get_project_root

def main():
    project_root = get_project_root()
    raw_data_file = Path(project_root) / "datasets" / "compas_scores" / "compas-scores-two-years.csv"
    raw_data = pd.read_csv(raw_data_file)

    # Drop target and leakage identifiers
    drop_cols = [
        'score_text',          # target
        'decile_score',        # direct numeric mapping of target
        'score_factor',        # derived features relating to target
        'id', 'name', 'c_case_number',  # unique identifiers
        'compas_screening_date', 'dob'  # date fields
    ]
    X = raw_data.drop(columns=[c for c in drop_cols if c in raw_data.columns])
    y = raw_data['score_text']

    # Separate feature types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Preprocessing pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Full pipeline
    clf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train and predict
    clf_pipeline.fit(X_train, y_train)
    y_pred = clf_pipeline.predict(X_test)

    # Evaluate
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
