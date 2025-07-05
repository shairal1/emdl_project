import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
raw_data = pd.read_csv(raw_data_file)

# Drop the target variable
X = raw_data.drop('score_text', axis=1)
y = raw_data['score_text']

# Split data into training and testing sets BEFORE any preprocessing to prevent data leakage
# MODIFICATION: Data split moved earlier to prevent data leakage.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encoding the target variable AFTER splitting
# MODIFICATION: Target encoding moved after split to prevent data leakage.
le_y = LabelEncoder()
y_train_encoded = le_y.fit_transform(y_train)
y_test_encoded = le_y.transform(y_test)

# Identify numerical and categorical columns for preprocessing
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Create preprocessing pipelines for numerical and categorical features
# MODIFICATION: Separate imputation strategies for numerical (median) and categorical (most_frequent) columns.
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Median for numerical, robust to outliers
    ('scaler', StandardScaler()) # MODIFICATION: Added StandardScaler for numerical features.
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Most frequent for categorical
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) # OrdinalEncoder for tree models
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough' # Keep other columns not explicitly transformed
)

# Create the full pipeline including preprocessing and the classifier
# MODIFICATION: Preprocessing steps (imputation, scaling, encoding) now encapsulated in a Pipeline to prevent data leakage by fitting only on training data.
clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(random_state=42))])

# Perform Cross-validation on the training set
# MODIFICATION: Added Stratified K-Fold Cross-validation for robust performance estimation.
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf_pipeline, X_train, y_train_encoded, cv=cv_stratified, scoring='accuracy')

print("Cross-validation Accuracy Scores:", cv_scores)
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train the model on the full training data and evaluate on the test set
clf_pipeline.fit(X_train, y_train_encoded)
y_pred_encoded = clf_pipeline.predict(X_test)

# Evaluate the model
print("\n--- Test Set Evaluation ---")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_encoded))
print("Classification report:")
print(classification_report(y_test_encoded, y_pred_encoded, target_names=le_y.classes_))

# Calculate ROC AUC score for binary classification
# MODIFICATION: Added ROC AUC score for comprehensive evaluation.
if len(le_y.classes_) == 2:
    y_proba = clf_pipeline.predict_proba(X_test)[:, 1] # Probability of the positive class
    roc_auc = roc_auc_score(y_test_encoded, y_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
else:
    print("ROC AUC score is typically used for binary classification. For multi-class, consider macro/weighted average.")