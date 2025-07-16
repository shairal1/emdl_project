#FIXED Summary of fixes:
#FIXED - Dropped 'score' column to prevent data leakage.
#FIXED - Added stratify=y to train_test_split for class balance.
#FIXED - Added SimpleImputer to handle missing categorical values before OneHotEncoder.
#FIXED - Set RandomForestClassifier n_jobs=-1 for parallelism.
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.experimental import enable_iterative_imputer  # noqa
#FIXED
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
raw_data = pd.read_csv(raw_data_file)

#FIXED
X = raw_data.drop(['score_text', 'score'], axis=1)
y = raw_data['score_text']

#FIXED
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

#FIXED
numeric_transformer = IterativeImputer(random_state=42)
#FIXED
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#FIXED
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    #FIXED
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')

print("Cross-validation scores:", scores)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
