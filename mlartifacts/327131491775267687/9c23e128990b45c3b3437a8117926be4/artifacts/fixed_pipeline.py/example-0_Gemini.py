import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Changed Normalizer to StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import re # Added for more robust text normalization

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

# [Incorrect Text Normalization] - Improved text normalization for 'occupation' column
# Removed non-alphabetic characters, lowercased, stripped whitespace, and consolidated multiple spaces
data['occupation'] = data['occupation'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
data['occupation'] = data['occupation'].str.lower().str.strip()
data['occupation'] = data['occupation'].str.replace(r'\s+', ' ', regex=True)

# [Incorrect Spatial Aggregation] - Removed the line that caused high information loss by replacing native-country with a single value
# The original 'native-country' values will now be handled by OneHotEncoder as part of categorical features
# data['native-country'] = data['native-country'].apply(lambda x: 'North America') # This line was removed

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(data.drop('salary', axis=1), data['salary'], test_size=0.2, random_state=42)

# Defining preprocessing for numeric and categorical features
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    # [Missing Data Imputation Strategy] - Kept 'median' as a reasonable imputation strategy for numerical features
    ('imputer', SimpleImputer(strategy='median')),
    # [Lack of Feature Scaling Consideration] - Changed Normalizer to StandardScaler for general feature scaling
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    # [Missing Data Imputation Strategy] - Changed imputation strategy to 'most_frequent' for categorical features
    ('imputer', SimpleImputer(strategy='most_frequent')),
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
    ('classifier', RandomForestClassifier())
])

# Fitting model
pipeline.fit(X_train, y_train)

# Evaluating the model
score = pipeline.score(X_test, y_test)
print(f"Model accuracy: {score:.2f}")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))