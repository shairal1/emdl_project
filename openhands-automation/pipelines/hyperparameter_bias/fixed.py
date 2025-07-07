# Summary of fixes:
# - Changed target variable from score_text to two_year_recid.
# - Dropped date and data leakage columns ('dob', 'c_jail_in', 'c_jail_out', 'is_recid').
# - Added stratify parameter to train_test_split.
# - Used join='outer' when aligning train and test features.
# - Increased n_estimators and removed max_depth limit for RandomForestClassifier.

import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
raw_data = pd.read_csv(raw_data_file)

#FIXED Remove out-of-range screening records
raw_data = raw_data[
    (raw_data['days_b_screening_arrest'] <= 30) &
    (raw_data['days_b_screening_arrest'] >= -30)
]

#FIXED Select relevant columns and drop dates/leakage columns
raw_data = raw_data[
    ['sex', 'age', 'c_charge_degree', 'race', 'priors_count',
     'days_b_screening_arrest', 'decile_score', 'two_year_recid']
]

# Fill missing values
for column in raw_data.columns:
    if raw_data[column].dtype == 'object':
        raw_data[column] = raw_data[column].fillna(raw_data[column].mode()[0])
    else:
        raw_data[column] = raw_data[column].fillna(raw_data[column].mean())

#FIXED Define features and target correctly
X = raw_data.drop(columns=['two_year_recid'])
#FIXED Define target variable
y = raw_data['two_year_recid']

#FIXED Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# One-hot encode categorical variables
categorical_cols = X_train.select_dtypes(include=['object']).columns
X_train = pd.get_dummies(X_train, columns=categorical_cols)
X_test = pd.get_dummies(X_test, columns=categorical_cols)

#FIXED Align features using outer join
X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

#FIXED Use stronger RandomForest settings
clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification report: {classification_report(y_test, y_pred)}")
