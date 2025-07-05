import sys
import os
import pandas as pd
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score # Modified: Added cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import numpy as np # Added: For cross_val_score mean/std

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

# Removed: Incorrect text normalization for 'occupation'
# data['occupation'] = data['occupation'].str.lower()
# data['occupation'] = data['occupation'].str.replace('-', ' ')

# Removed: Incorrect spatial aggregation for 'native-country'
# data['native-country'] = data['native-country'].apply(lambda x: 'North America')

# Added: Feature Engineering - Creating 'age_group' feature
# This is a deterministic transformation, safe to do before train-test split.
data['age_group'] = pd.cut(data['age'],
                           bins=[0, 18, 25, 35, 45, 55, 65, np.inf], # Using np.inf for the upper bound
                           labels=['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
                           right=False, # Bins are [min, max)
                           include_lowest=True) # Include the lowest value

# Splitting data
X = data.drop('salary', axis=1)
y = data['salary']
# Modified: Kept train_test_split for final evaluation comparison, but added cross-validation for robust assessment.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Added random_state for reproducibility

# Defining preprocessing for numeric and categorical features
# Features are selected based on X_train after feature engineering
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns # Modified: Added 'category' for age_group

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('normalizer', Normalizer())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Combining preprocessing with the classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42)) # Added random_state for reproducibility
])

# Removed: No hyperparameter tuning as per instructions.
# No changes to RandomForestClassifier defaults other than random_state.

# Added: Cross-Validation for robust performance estimation
# The pipeline structure correctly prevents data leakage as transformations are fit on training folds only.
print("Performing Cross-Validation...")
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1) # Using X and y for full dataset CV
print(f"Cross-Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
print(f"All CV Scores: {cv_scores}")

# Fitting model on the entire X_train (from initial split) for final evaluation
print("\nFitting model on training data for final test set evaluation...")
pipeline.fit(X_train, y_train)

# Evaluating the model on the held-out test set
score = pipeline.score(X_test, y_test)
print(f"Model accuracy on test set: {score:.2f}")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))