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
import numpy as np # Added: Imported numpy for mean and std deviation calculation

# Setting up paths

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root,"LLM_automation","test_pipeline","adult_data.csv")
data = pd.read_csv(raw_data_file)

data['occupation'] = data['occupation'].str.lower()  
data['occupation'] = data['occupation'].str.replace('-', ' ')  
# [Data Leakage] and [Improper handling of categorical features] addressed:
# Removed the incorrect spatial aggregation for 'native-country'.
# The original 'native-country' values will now be processed by OneHotEncoder.
# This also indirectly addresses a consequence of [Inadequate Data Exploration],
# as proper exploration would reveal the diversity and potential importance of 'native-country' values.

# Splitting data
X = data.drop('salary', axis=1)
y = data['salary']
# Modified: Added random_state for reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining preprocessing for numeric and categorical features
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

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
    # Modified: Added random_state for reproducibility of the classifier
    ('classifier', RandomForestClassifier(random_state=42)) 
])

# [Lack of Cross-Validation] addressed:
# Perform cross-validation on the training data to get a robust estimate of model performance.
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1) 
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.2f} +/- {np.std(cv_scores):.2f}")

# Fit the pipeline on the full training data
pipeline.fit(X_train, y_train)

# Evaluating the model on the hold-out test set
score = pipeline.score(X_test, y_test)
print(f"Model accuracy on test set: {score:.2f}")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))