import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold # Modified: Added cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler # Modified: Added StandardScaler
from sklearn.impute import SimpleImputer # Modified: Added SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

# Modified: Handle potential missing values (often represented as '?' in the adult dataset) by replacing them with NaN
# This ensures SimpleImputer can correctly identify and handle them.
data.replace('?', np.nan, inplace=True)

# Splitting the data
X = data.drop('salary', axis=1)
y = data['salary']

# Defining categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['number']).columns.tolist()

# Modified: Creating robust preprocessing pipelines for numerical and categorical features
# This addresses Missing_data_handling and Lack_of_scaling.
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Impute missing numerical values with the mean
    ('scaler', StandardScaler()) # Scale numerical features (addresses Lack_of_scaling)
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing categorical values with the most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features, ignore unknown categories for robustness
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Creating a pipeline that first transforms data and then fits the model
# Modified: Added class_weight='balanced' to handle potential class imbalance (addresses Unbalanced_data)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(class_weight='balanced', random_state=42)) # Added random_state for reproducibility
])

# Splitting the data for initial training, testing, and noise sensitivity test
# Modified: Added stratify=y to ensure the train/test split maintains the same proportion of classes as in the original dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Training the model on the train split
pipeline.fit(X_train, y_train)

# Evaluating the model on the clean test set
y_pred = pipeline.predict(X_test)
accuracy_before_noise = accuracy_score(y_test, y_pred)

# Introducing more significant Gaussian noise to the test set (numerical features only)
X_test_noisy = X_test.copy()
numerical_indices_test_set = X_test_noisy.select_dtypes(include=['float64', 'int64']).columns
X_test_noisy[numerical_indices_test_set] += np.random.normal(0, 15, X_test_noisy[numerical_indices_test_set].shape)

# Evaluating the model on noisy data
y_pred_noisy = pipeline.predict(X_test_noisy)
accuracy_after_noise = accuracy_score(y_test, y_pred_noisy)

print(f"Accuracy before noise (on clean test set): {accuracy_before_noise:.4f}")
print(f"Accuracy after noise (on noisy test set): {accuracy_after_noise:.4f}")

# Modified: Perform cross-validation for more robust model evaluation (addresses Lack_of_cross_validation)
# Using StratifiedKFold to maintain class proportions across folds, which is important for imbalanced datasets.
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 5-fold cross-validation with shuffling
cv_scores = cross_val_score(pipeline, X, y, cv=cv_stratified, scoring='accuracy', n_jobs=-1) # Use all available CPU cores

print("\n--- Cross-Validation Results ---")
print(f"Cross-validation accuracies: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard deviation of CV accuracy: {np.std(cv_scores):.4f}")

# Modified: Perform Feature Importance Analysis (addresses Feature_importance_analysis)
print("\n--- Feature Importance Analysis ---")
# Get feature names after preprocessing from the fitted preprocessor
preprocessor_fitted = pipeline.named_steps['preprocessor']
feature_names_out = preprocessor_fitted.get_feature_names_out()

# Get feature importances from the trained RandomForestClassifier model within the pipeline
model = pipeline.named_steps['model']
feature_importances = model.feature_importances_

# Create a DataFrame to display feature importances for better readability
importance_df = pd.DataFrame({'Feature': feature_names_out, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df.head(10)) # Display top 10 most important features
print(f"\nTotal number of features after one-hot encoding: {len(feature_names_out)}")