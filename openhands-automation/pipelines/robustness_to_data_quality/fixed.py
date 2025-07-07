# Summary of fixes:
# - Added handle_unknown='ignore' and sparse=False to OneHotEncoder to handle unseen categories and produce dense output (#FIXED)
# - Set random_state=42 for RandomForestClassifier for reproducible results (#FIXED)
# - Seeded numpy's random number generator before noise addition for reproducibility (#FIXED)
# - Changed dtypes selection for numeric columns to include all numeric types (#FIXED)
# - Used .loc indexing to avoid SettingWithCopyWarning when adding noise (#FIXED)

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

X = data.drop('salary', axis=1)
y = data['salary']

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['number']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        #FIXED: updated OneHotEncoder to ignore unknown categories and output dense array
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features),
        ('num', 'passthrough', numerical_features)
    ],
    remainder='drop'  # Drop any other columns not specified
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    #FIXED: added random_state for reproducible results
    ('model', RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy_before_noise = accuracy_score(y_test, y_pred)

X_test_noisy = X_test.copy()
#FIXED: seeded numpy RNG for noise reproducibility
np.random.seed(42)
#FIXED: changed to include all numeric types for noise addition
numerical_indices = X_test_noisy.select_dtypes(include=['number']).columns
#FIXED: using .loc to avoid SettingWithCopyWarning when adding noise
X_test_noisy.loc[:, numerical_indices] += np.random.normal(0, 15, X_test_noisy[numerical_indices].shape)  

y_pred_noisy = pipeline.predict(X_test_noisy)
accuracy_after_noise = accuracy_score(y_test, y_pred_noisy)

print(f"Accuracy before noise: {accuracy_before_noise}")
print(f"Accuracy after noise: {accuracy_after_noise}")
