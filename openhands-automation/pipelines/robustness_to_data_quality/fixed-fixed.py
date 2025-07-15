# Summary of fixes:
# 1. Replaced "?" entries with NaN for proper missing value handling.
# 2. Added SimpleImputer for categorical features before OneHotEncoder.
# 3. Set OneHotEncoder to sparse=False for compatibility with RandomForestClassifier.
# 4. Expanded add_noise to include all numeric types via np.number.
# 5. Unified noise application for train and test using add_noise.
# 6. Seeded numpy random generator for reproducibility.
# 7. Added n_jobs=-1 to RandomForestClassifier for parallelism.

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

#FIXED
data.replace("?", np.nan, inplace=True)

X = data.drop('salary', axis=1)
y = data['salary']

#FIXED
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
#FIXED
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  #FIXED
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))        #FIXED
        ]), categorical_features),
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ]), numerical_features)
    ],
    remainder='drop'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))  #FIXED
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def add_noise(X, noise_level=0.1):  #FIXED
    noisy_data = X.copy()
    #FIXED
    numerical_cols = noisy_data.select_dtypes(include=[np.number]).columns
    noisy_data[numerical_cols] += np.random.normal(0, noise_level, noisy_data[numerical_cols].shape)
    return noisy_data

#FIXED
np.random.seed(42)

X_train_noisy = add_noise(X_train, noise_level=0.1)
pipeline.fit(X_train_noisy, y_train)

y_pred = pipeline.predict(X_test)
accuracy_before_noise = accuracy_score(y_test, y_pred)

#FIXED
X_test_noisy = add_noise(X_test, noise_level=0.1)

y_pred_noisy = pipeline.predict(X_test_noisy)
accuracy_after_noise = accuracy_score(y_test, y_pred_noisy)

print(f"Accuracy before noise: {accuracy_before_noise}")
print(f"Accuracy after noise: {accuracy_after_noise}")