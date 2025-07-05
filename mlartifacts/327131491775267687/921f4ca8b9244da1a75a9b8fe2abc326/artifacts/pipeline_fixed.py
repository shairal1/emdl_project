```python
# All imports
import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin # Added for custom transformer

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
raw_data = pd.read_csv(raw_data_file)

# Data Splitting before Data Preparation
train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)

# Data Extraction
columns_to_use = ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
                  'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']

train_data = train_data[columns_to_use]
test_data = test_data[columns_to_use]

print("Shape of training data:", train_data.shape)
print("Shape of testing data:", test_data.shape)

# Data Replacement
train_data = train_data.replace('Medium', "Low")
test_data = test_data.replace('Medium', "Low")

# Binarizing labels
train_labels = label_binarize(train_data['score_text'], classes=['High', 'Low']).ravel()
test_labels = label_binarize(test_data['score_text'], classes=['High', 'Low']).ravel()

# Dropping the 'score_text' column from features
train_data = train_data.drop(columns=['score_text'])
test_data = test_data.drop(columns=['score_text'])

# Resetting indices to avoid alignment issues
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# MODIFICATION START: Address [Date Handling] and [Inconsistent Feature Handling], [Missing Data Handling], [Potential Data Imbalance]

# Custom Transformer for Date Feature Engineering
# This addresses [Date Handling] by converting date strings to datetime objects
# and extracting numerical features, then dropping the original date columns.
class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols=None):
        self.date_cols = date_cols if date_cols is not None else ['dob', 'c_jail_in', 'c_jail_out']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # Convert specified date columns to datetime objects, coercing errors
        for col in self.date_cols:
            X_copy[col] = pd.to_datetime(X_copy[col], errors='coerce')

        # Calculate 'jail_duration_days' as the difference between jail_out and jail_in
        X_copy['jail_duration_days'] = (X_copy['c_jail_out'] - X_copy['c_jail_in']).dt.days
        # Replace NaT (Not a Time) results with NaN for numerical imputation
        X_copy['jail_duration_days'] = X_copy['jail_duration_days'].replace({pd.NaT: pd.NA}).astype(float)

        # Drop the original date columns after engineering features from them
        X_copy = X_copy.drop(columns=self.date_cols)
        
        return X_copy

# Define feature groups for comprehensive and consistent preprocessing
# This addresses [Inconsistent Feature Handling] and [Missing Data Handling] by ensuring
# all relevant features are processed and missing values are handled.

# Features to be scaled after imputation (includes the new derived date feature)
numerical_features_scale = ['priors_count', 'days_b_screening_arrest', 'decile_score', 'jail_duration_days']

# Features to be One-Hot Encoded after imputation
categorical_features_onehot = ['sex', 'c_charge_degree', 'race', 'is_recid', 'two_year_recid']

# 'age' is kept separate as it was originally binned
age_feature = ['age']

# Create preprocessing pipelines for different feature types
# Numerical features: impute with median (robust to outliers), then scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Addresses [Missing Data Handling] for numerical
    ('scaler', StandardScaler())
])

# Categorical features: impute with most frequent, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Addresses [Missing Data Handling] for categorical
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Age feature: impute with mean, then bin (as per original logic)
age_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Addresses [Missing Data Handling] for age
    ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))
])

# Create a ColumnTransformer to apply different transformations to different columns
# This comprehensively addresses [Inconsistent Feature Handling] by processing all relevant features.
featurizer = ColumnTransformer(
    transformers=[
        ('age_bin', age_transformer, age_feature),
        ('num_scale', numerical_transformer, numerical_features_scale),
        ('cat_onehot', categorical_transformer, categorical_features_onehot)
    ],
    remainder='drop' # Drop any columns not specified to avoid unintended input to the model
)

# Complete pipeline with feature engineering, preprocessing, and classification
# The entire pipeline is fitted on train_data only, which correctly handles [Data Leakage].
pipeline = Pipeline(steps=[
    ('date_engineer', DateFeatureEngineer()), # Step 1: Custom date feature engineering
    ('featurizer', featurizer),              # Step 2: Apply all other preprocessing using ColumnTransformer
    ('classifier', LogisticRegression(
        solver='liblinear', # Recommended solver for smaller datasets or L1 regularization
        random_state=42,
        class_weight='balanced' # Addresses [Potential Data Imbalance]
    ))
])

# Training the pipeline
# All preprocessing steps (including imputation and scaling) learn parameters ONLY from train_data.
# This ensures no data leakage from the test set.
pipeline.fit(train_data, train_labels)

# Model Evaluation
print("Model score:", pipeline.score(test_data, test_labels))

# Classification Report
print(classification_report(test_labels, pipeline.predict(test_data), zero_division=0))

# MODIFICATION END
```