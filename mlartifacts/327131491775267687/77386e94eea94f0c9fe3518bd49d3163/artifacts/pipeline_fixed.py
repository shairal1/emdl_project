```python
# All imports
import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

# Data preparation steps. Note: Data has not been split yet
# Data Extraction
raw_data = raw_data[
    ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
     'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

# Data Filtering
raw_data = raw_data[(raw_data['days_b_screening_arrest'] <= 30) & (raw_data['days_b_screening_arrest'] >= -30)]
raw_data = raw_data[raw_data['is_recid'] != -1]
raw_data = raw_data[raw_data['c_charge_degree'] != "O"]
raw_data = raw_data[raw_data['score_text'] != 'N/A']

# Data Replacement
raw_data = raw_data.replace('Medium', "Low")

# MODIFICATION START: Define features (X) and target (y) before splitting and transformation.
# This clarifies the target variable usage and ensures consistent data preparation for features.
X = raw_data.drop('score_text', axis=1) # Features
y = raw_data['score_text']             # Target

# MODIFICATION: Perform train/test split BEFORE fitting any transformers to prevent data leakage.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Preparation Pipeline (Imputation, Encoding, Discretization)
impute1_and_onehot = Pipeline([('imputer1', SimpleImputer(strategy='most_frequent')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])
impute2_and_bin = Pipeline([('imputer2', SimpleImputer(strategy='mean')),
                            ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))])

featurizer = ColumnTransformer(transformers=[
    ('impute1_and_onehot', impute1_and_onehot, ['is_recid']),
    ('impute2_and_bin', impute2_and_bin, ['age'])
])

# MODIFICATION: Fit the featurizer ONLY on the training data.
X_train_prepared = featurizer.fit_transform(X_train)

# MODIFICATION: Transform the test data using the *fitted* featurizer (do not fit on test data).
X_test_prepared = featurizer.transform(X_test)

print("Shape of training data:", X_train_prepared.shape)
print("Shape of testing data:", X_test_prepared.shape)

# Binarizing labels
# Given 'Medium' was replaced with 'Low', the target classes are effectively 'High' and 'Low',
# making label_binarize appropriate for a binary classification problem.
y_train_binarized = label_binarize(y_train, classes=['High', 'Low'])
y_test_binarized = label_binarize(y_test, classes=['High', 'Low'])

# Model Evaluation
# MODIFICATION: Added max_iter to LogisticRegression to help with convergence warnings.
pipeline = Pipeline([('classifier', LogisticRegression(max_iter=1000))])

# Fit the model using the prepared training data and binarized training labels.
# .ravel() is used to convert the 2D output of label_binarize to a 1D array, as expected by LogisticRegression for binary targets.
pipeline.fit(X_train_prepared, y_train_binarized.ravel())
print("Accuracy", pipeline.score(X_test_prepared, y_test_binarized.ravel()))

# Classification Report
print(classification_report(y_test_binarized, pipeline.predict(X_test_prepared), zero_division=0))
# MODIFICATION END
```