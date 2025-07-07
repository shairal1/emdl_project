# Summary of fixes:
# - Removed dependency on utils.get_project_root and unused import sys; compute project_root via os.path. #FIXED
# - Treated '?' as missing values in read_csv via na_values=['?']. #FIXED
# - Included 'category' dtype in categorical_cols selection and used 'number' include for numeric_cols selection. #FIXED
# - Set OneHotEncoder to sparse=False to output dense arrays. #FIXED
# - Set n_jobs=-1 in RandomForestClassifier for parallel training. #FIXED
# - Added __main__ guard and improved printing of classification report. #FIXED

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Compute project root instead of using utils.get_project_root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))  #FIXED

raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
# Treat '?' as missing values
data = pd.read_csv(raw_data_file, na_values=['?'])  #FIXED

X = data.drop(columns=['salary'])
y = data['salary']

# Include 'category' dtype and use 'number' include for numeric columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns  #FIXED
numeric_cols = X.select_dtypes(include=['number']).columns  #FIXED

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))  #FIXED
        ]), categorical_cols)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))  #FIXED
])

def main():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification report:")  #FIXED
    print(classification_report(y_test, y_pred, zero_division=0))  #FIXED

if __name__ == "__main__":
    main()  #FIXED
