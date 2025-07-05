import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# [Data Leakage], [Target Leakage], [Feature Selection/Engineering], [Handling of Date Features]
# Removed 'c_jail_in' and 'c_jail_out' due to data leakage (future information).
# Removed 'decile_score' due to target leakage (directly derived from COMPAS and highly correlated with score_text).
# Removed 'dob' as 'age' is already present and used, making 'dob' redundant for this simple pipeline.
raw_data = raw_data[
    ['sex', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
     'is_recid', 'two_year_recid']]

# Filling missing values
for column in raw_data.columns:
    if raw_data[column].dtype == 'object':
        raw_data[column] = raw_data[column].fillna(raw_data[column].mode()[0])
    else:
        raw_data[column] = raw_data[column].fillna(raw_data[column].mean())

X = raw_data.drop(columns=['score_text'])
y = raw_data['score_text']

# [Class Imbalance]
# Check for class imbalance and apply class weighting if necessary.
# print(y.value_counts()) # For inspection, typical output shows imbalance: Low > High > Medium
# The RandomForestClassifier can handle imbalance using class_weight.

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identifying categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Converting categorical columns to numeric using one-hot encoding on training data
X_train = pd.get_dummies(X_train, columns=categorical_cols)

# Ensuring the same columns are present in the test set
X_test = pd.get_dummies(X_test, columns=categorical_cols)

# Aligning the test set to the training set to ensure consistent columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Training a classifier with arbitrary hyper-parameters
# [Class Imbalance] - Added class_weight='balanced' to address potential class imbalance in the target variable.
clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# Prediction and evaluation
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification report: {classification_report(y_test, y_pred)}")

# [Lack of Data Exploration], [Bias Consideration], [Limited Model Evaluation]
# Note: In a full pipeline, extensive EDA would precede these steps.
# Bias evaluation across sensitive attributes (like 'race', 'sex') and
# a more comprehensive model evaluation (e.g., ROC AUC, precision-recall curves, confusion matrix, fairness metrics)
# would be critical for a high-stakes domain like criminal justice.