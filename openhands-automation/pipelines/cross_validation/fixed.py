# fixed.py

"""Summary of fixes:

1. Prevented data leakage by moving undersampling and feature selection inside the cross-validation pipeline.
2. Removed manual resampling and feature selection outside cross-validation.
3. Removed unused imports (train_test_split).
4. Introduced a reproducible StratifiedKFold with shuffle and random_state.
5. Printed mean and standard deviation of cross-validation scores.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


def main():
    # Set up project root path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
    # Import project-specific utility
    from utils import get_project_root
    project_root = get_project_root()

    # Load dataset
    raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "5050_split.csv")
    data = pd.read_csv(raw_data_file)

    # Separate features and target
    X = data.drop('Diabetes_binary', axis=1)
    y = data['Diabetes_binary']

    # Create pipeline: undersampling -> feature selection -> classifier
    pipeline = Pipeline([
        ('under', RandomUnderSampler(random_state=42)),
        ('select', SelectKBest(f_classif, k=10)),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # Use stratified K-fold for reproducibility
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv)

    # Report results
    print("Cross-validation scores:", scores)
    print(f"Mean CV score: {scores.mean():.3f}      ± {scores.std():.3f}")


if __name__ == "__main__":
    main()
