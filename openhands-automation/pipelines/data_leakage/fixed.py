# Summary of fixes:
# 1. Handled missing values represented as '?' via na_values in read_csv.
# 2. Stripped whitespace from categorical features and target to ensure consistency.
# 3. Applied constant 'missing' imputation for categorical missing values.
# 4. Used stratify=y in train_test_split to maintain class distribution.
# 5. Set random_state=42 and increased max_iter in LogisticRegression for reproducibility and convergence.
# 6. Encapsulated execution in a main() function and added __main__ guard.

import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils import get_project_root

def main():
    # Determine project root and data path
    project_root = get_project_root()
    raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")

    # Read data, treat '?' as missing values
    data = pd.read_csv(raw_data_file, na_values=[' ?', '?'])

    # Strip whitespace from all object (string) columns
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].str.strip()

    # Define feature columns and target
    numeric_columns = ['age', 'hours-per-week']
    categorical_columns = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    target = 'salary'

    # Preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

    # Split data into features and target
    X = data[numeric_columns + categorical_columns]
    y = data[target]

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split into training and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Build the full pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # Train and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

if __name__ == '__main__':
    main()