```python
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score # MODIFIED: Added cross_val_score for robust evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler # MODIFIED: Added StandardScaler for numerical feature scaling
from sklearn.compose import ColumnTransformer # MODIFIED: Added ColumnTransformer to streamline preprocessing
from sklearn.pipeline import Pipeline # MODIFIED: Added Pipeline for a cleaner and more robust workflow

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# The 'get_project_root' function itself does not cause data leakage.
# Data leakage typically arises from using information from the test set or future data during training/preprocessing.
from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

# Splitting the raw data into training and test sets
X = data.drop('salary', axis=1)
y = data['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODIFIED: Separate numerical and categorical features
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# MODIFIED: Create a preprocessor using ColumnTransformer
# [Unclear Feature Scaling] - StandardScaler is added for numerical features.
# This setup also inherently addresses potential data leakage by fitting transformers only on training data.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features), # Scale numerical features
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features) # One-hot encode categorical features
    ],
    remainder='passthrough' # Keep any other columns that are not explicitly transformed
)

# MODIFIED: Create a pipeline that first preprocesses the data and then trains the model
# This ensures all preprocessing steps are applied consistently (fit on train, transform on test).
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, C=1.0)) # [Fixed Regularization Strength] - C set to default (1.0). Hyperparameter tuning is explicitly excluded.
])

# Training the model
model_pipeline.fit(X_train, y_train)

# Evaluating the model on the held-out test set
y_pred = model_pipeline.predict(X_test)
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred))

# MODIFIED: [Limited Model Evaluation] - Add cross-validation for more robust evaluation
print("\nCross-validation scores (accuracy) on Training Set:")
cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"CV Scores: {cv_scores}")
```