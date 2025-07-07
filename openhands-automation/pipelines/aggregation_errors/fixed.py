# Summary of fixes:
# - Handle missing values and whitespace when reading CSV by specifying na_values and skipinitialspace in pd.read_csv.
# - Removed naive mapping of `native-country` to a single category; instead, normalize hyphens and lowercase.
# - Specified regex=False in str.replace to avoid future warnings.
# - Added random_state and stratify to train_test_split for reproducibility and balanced class splits.
# - Replaced Normalizer with StandardScaler for appropriate numeric feature scaling.
# - Set random_state in RandomForestClassifier for reproducible results.

import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder  #FIXED replaced Normalizer with StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file, na_values="?", skipinitialspace=True)  #FIXED handle missing values and whitespace

# Preprocess occupation
data['occupation'] = data['occupation'].str.lower()
data['occupation'] = data['occupation'].str.replace("-", " ", regex=False)  #FIXED specify regex=False to avoid warnings

# Preprocess native-country
data['native-country'] = data['native-country'].str.lower().str.replace("-", " ", regex=False)  #FIXED normalized hyphens and lowercase instead of naive mapping

# Split data with reproducibility and stratification
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("salary", axis=1),
    data["salary"],
    test_size=0.2,
    random_state=42,
    stratify=data["salary"],
)  #FIXED added random_state and stratify for reproducibility

# Identify numeric and categorical features
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

# Create preprocessing pipelines
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),  #FIXED use StandardScaler instead of Normalizer
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Build and train the pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),  #FIXED set random_state for reproducibility
    ]
)

pipeline.fit(X_train, y_train)

# Evaluate the model
score = pipeline.score(X_test, y_test)
print(f"Model accuracy: {score:.2f}")

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
