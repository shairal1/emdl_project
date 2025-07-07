# Summary of fixes:
# - Added skipinitialspace=True to pd.read_csv to remove leading spaces after delimiters
# - Replaced '?' with np.nan and dropped rows with missing target values
# - Stripped whitespace from all categorical and target columns
# - Converted numeric feature columns to numeric types to coerce invalid entries
# - Used stratify in train_test_split to maintain class distribution
# - Specified solver, max_iter, and random_state in LogisticRegression for convergence and reproducibility
# - Added zero_division parameter in classification_report to avoid division by zero errors

import pandas as pd
import numpy as np  #FIXED
import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file, skipinitialspace=True)  #FIXED

# Handle missing values represented by '?' and drop rows missing the target
data.replace('?', np.nan, inplace=True)  #FIXED

target = 'salary'

# Drop rows with missing target
data.dropna(subset=[target], inplace=True)  #FIXED

numeric_columns = ['age', 'hours-per-week']
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'native-country']

# Strip whitespace from categorical and target columns
for col in categorical_columns + [target]:
    data[col] = data[col].str.strip()  #FIXED

# Ensure numeric columns are numeric
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')  #FIXED

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

X = data[numeric_columns + categorical_columns]
y = data[target]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Stratify to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)  #FIXED

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42))  #FIXED
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred,
                            target_names=label_encoder.classes_,
                            zero_division=0))  #FIXED
