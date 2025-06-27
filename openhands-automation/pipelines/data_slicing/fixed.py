# fixed.py
# 
# Summary of fixes:
# - Replaced label_binarize with LabelEncoder to correctly encode the binary target as a 1D array.
# - Restricted replacement of 'Medium' to the 'score_text' column only, avoiding unintended replacements elsewhere.
# - Removed the redundant 'dob' and target-leaking 'two_year_recid' columns from the feature set.
# - Added full preprocessing for all features using ColumnTransformer:
#     * Numeric features: SimpleImputer(strategy='mean') + StandardScaler()
#     * Categorical features: SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')
# - Cleaned up imports (removed unused label_binarize) and increased max_iter for LogisticRegression to ensure convergence.

import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ensure parent directory is on path for utils import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# load data
project_root = get_project_root()
raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
raw_data = pd.read_csv(raw_data_file)

# select relevant columns and copy
columns_to_use = [
    'sex', 'age', 'c_charge_degree', 'race', 'priors_count',
    'days_b_screening_arrest', 'decile_score', 'is_recid',
    'c_jail_in', 'c_jail_out', 'score_text'
]
data = raw_data[columns_to_use].copy()

# map 'Medium' risk to 'Low' in score_text only
data['score_text'] = data['score_text'].replace('Medium', 'Low')

# split into train/test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# encode target
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_data['score_text'])
test_labels = label_encoder.transform(test_data['score_text'])

# drop target from feature sets
train_data = train_data.drop(columns=['score_text'])
test_data = test_data.drop(columns=['score_text'])

# define features by type
categorical_features = ['sex', 'c_charge_degree', 'race', 'is_recid']
numeric_features = [
    'age', 'priors_count', 'days_b_screening_arrest',
    'decile_score', 'c_jail_in', 'c_jail_out'
]

# pipelines for preprocessing
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# build and train full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline.fit(train_data, train_labels)

# evaluate
accuracy = pipeline.score(test_data, test_labels)
print(f"Model accuracy: {accuracy:.4f}")
print(classification_report(
    test_labels,
    pipeline.predict(test_data),
    zero_division=0,
    target_names=label_encoder.classes_
))
