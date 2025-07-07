# Summary of fixes:
# 1. Handled missing values: replaced '?' with NaN and dropped missing rows.
# 2. Added StandardScaler for numeric features.
# 3. Configured OneHotEncoder to output dense array (sparse=False).
# 4. Added stratify=y in train_test_split for reproducibility.
# 5. Added random_state=42 to LogisticRegression for reproducibility.

import os
import sys
import pandas as pd
#FIXED
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
#FIXED
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

#FIXED
# Handle missing values
#FIXED
data.replace('?', np.nan, inplace=True)
#FIXED
data.dropna(inplace=True)

X = data.drop('salary', axis=1)
y = data['salary']

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
#FIXED
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

#FIXED
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols)
    ],
    remainder='passthrough'  # Leave the rest of the columns untouched
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    #FIXED
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

#FIXED
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
