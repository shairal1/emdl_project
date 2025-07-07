# Summary of fixes:
# - Added Pipeline and ColumnTransformer for preprocessing of numeric and categorical features
# - Numeric features imputed with median, categorical with most frequent and OneHotEncoded to avoid ordinal encoding
# - Removed manual LabelEncoder for features and target, relying on classifier for label encoding
# - Included stratify=y in train_test_split to preserve class distribution
# - Wrapped script execution in main() under if __name__ guard
# - Improved print format for classification report

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
#FIXED from sklearn.preprocessing import OneHotEncoder
#FIXED from sklearn.pipeline import Pipeline
#FIXED from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

#FIXED def main():
def main():
    project_root = get_project_root()
    raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
    raw_data = pd.read_csv(raw_data_file)

    X = raw_data.drop('score_text', axis=1)
    y = raw_data['score_text']

    #FIXED numeric and categorical feature lists
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    #FIXED detect objects and categories for categorical
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    #FIXED pipelines for numeric and categorical
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    #FIXED create ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    #FIXED classifier pipeline combining preprocessing and model
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    #FIXED stratify parameter in train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #FIXED fit and predict using the pipeline
    clf_pipeline.fit(X_train, y_train)
    y_pred = clf_pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

#FIXED add main guard
if __name__ == "__main__":
    main()