# Summary of fixes:
# - Added stratify=y to train_test_split to maintain class distribution
# - Added SimpleImputer to pipeline to handle missing values
# - Moved scaler after feature_selection in pipeline
# - Increased max_iter for LogisticRegression to ensure convergence
# - Added n_jobs=-1 to RandomForestClassifier and cross_val_score for parallel processing
# - Formatted accuracy outputs for readability

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np
#FIXED
from sklearn.impute import SimpleImputer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "binary_health_indicators.csv")
data = pd.read_csv(raw_data_file)

X = data.drop(columns=['Diabetes_binary'])
y = data['Diabetes_binary']

#FIXED
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#FIXED
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('feature_selection', SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        threshold="mean"
    )),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
#FIXED
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

#FIXED
cv_scores = cross_val_score(pipeline, X, y, cv=5, n_jobs=-1)
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
#FIXED
print(f"Mean Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")
