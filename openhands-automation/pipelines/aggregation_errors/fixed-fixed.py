"""
Summary of fixes:
- Removed unused import FunctionTransformer.
- Added regex=False to str.replace to avoid future warnings.
- Added stratify parameter to train_test_split to maintain class distribution.
- Configured OneHotEncoder to output dense arrays (sparse=False) to ensure compatibility with SMOTE.
- Set random_state in RandomForestClassifier for reproducibility.
- Updated print statement for model accuracy to properly format f-string.
"""

import os
import sys
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
#FIXED
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

def text_preprocessing(text_series):
    text_series = text_series.str.lower()
    #FIXED
    text_series = text_series.str.replace('-', ' ', regex=False)
    return text_series

def spatial_aggregation(location_series):
    location_series = location_series.apply(lambda x: 'North America' if x in ['United-States', 'Canada', 'Mexico'] else x)
    return location_series  

data['occupation'] = text_preprocessing(data['occupation'])
data['native-country'] = spatial_aggregation(data['native-country'])

#FIXED
X_train, X_test, y_train, y_test = train_test_split(data.drop('salary', axis=1), data['salary'], test_size=0.2, random_state=42, stratify=data['salary'])

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    #FIXED
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    #FIXED
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)

#FIXED
print(f"Model accuracy: {pipeline.score(X_test, y_test):.4f}")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))