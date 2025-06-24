# fixed.py
# Summary of fixes:
# - Restricted replacement of 'Medium' to only the 'score_text' column to avoid unintended replacements.
# - Dropped the 'dob' column as it's non-numeric and redundant with 'age'.
# - Replaced label_binarize with LabelEncoder for correct 1D label encoding.
# - Ensured labels are 1D arrays to prevent shape mismatches in LogisticRegression and classification_report.
# - Configured OneHotEncoder to return dense arrays (sparse=False) for compatibility in the feature matrix.

import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Handle project root discovery
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

# Load raw data
raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
raw_data = pd.read_csv(raw_data_file)

# Select relevant columns and make a copy
cols = [
    'sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text',
    'priors_count', 'days_b_screening_arrest', 'decile_score',
    'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out'
]
raw_data = raw_data[cols].copy()

# Replace 'Medium' risk labels with 'Low' only in the score_text column
raw_data['score_text'] = raw_data['score_text'].replace('Medium', 'Low')

# Drop non-numeric or redundant columns
raw_data = raw_data.drop(columns=['dob'])

# Define feature transformations
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))
])

# Build ColumnTransformer
featurizer = ColumnTransformer(
    transformers=[
        ('cat_recid', cat_pipeline, ['is_recid']),
        ('num_age', num_pipeline, ['age'])
    ],
    remainder='drop'  # other columns are dropped in this simplified pipeline
)

# Prepare features and labels
X = featurizer.fit_transform(raw_data.drop(columns=['score_text']))

# Encode labels as binary (Low=0, High=1)
le = LabelEncoder()
raw_data['label'] = le.fit_transform(raw_data['score_text'])  # Low -> 0, High -> 1
y = raw_data['label'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Shape of training data:", X_train.shape)
print("Shape of testing data:", X_test.shape)

# Train logistic regression classifier
pipeline = Pipeline([
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)

# Evaluate
accuracy = pipeline.score(X_test, y_test)
print("Accuracy", accuracy)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))