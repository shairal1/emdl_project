# Fixed pipeline script
# Summary of fixes:
# - Imported numpy as np to handle NaN placeholder (#FIXED)
# - Replaced placeholder '?' in the dataset with np.nan (#FIXED)
# - Added stratify parameter to train_test_split for balanced splits (#FIXED)
# - Specified solver='liblinear', random_state=42, and max_iter=1000 for LogisticRegression to ensure convergence (#FIXED)

import pandas as pd
#FIXED: Imported numpy to handle NaN values
import numpy as np
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
data = pd.read_csv(raw_data_file)
#FIXED: Replaced missing placeholder '?' with np.nan for proper imputation
data = data.replace(' ?', np.nan)

numeric_columns = ['age', 'hours-per-week']
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

target = 'salary'

X = data[numeric_columns + categorical_columns]
y = data[target]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#FIXED: Added stratify parameter to train_test_split for balanced target splits
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

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

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    #FIXED: Specified solver, max_iter, and random_state for LogisticRegression to ensure convergence
    ('classifier', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
