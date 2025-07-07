# Summary of fixes:
# - Dropped rows with missing values to avoid NaNs in train/test splits
# - Added stratify=y in train_test_split to maintain class distribution
# - Replaced OneHotEncoder parameter sparse_output with sparse for compatibility
# - Added zero_division=0 to classification_report to handle zero-division warnings

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "5050_split.csv")
data = pd.read_csv(raw_data_file)
#FIXED
# Drop rows with missing values to prevent errors during encoding or modeling
data = data.dropna()

print("Raw data gender distribution:\n", data['Sex'].value_counts(normalize=True).round(2))

data_filtered = data[data['Age'] > 4]
data_filtered = data_filtered[data_filtered['HighChol'] > 0]

X = data_filtered.drop('Diabetes_binary', axis=1)
y = data_filtered['Diabetes_binary']
#FIXED
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Maintain class distribution in splits
)

print("Test set gender distribution:\n", X_test['Sex'].value_counts(normalize=True).round(2))

#FIXED
# Use 'sparse' instead of 'sparse_output' for compatibility
encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train.select_dtypes(include=['object']))
X_test_encoded = encoder.transform(X_test.select_dtypes(include=['object']))

X_train_encoded_df = pd.DataFrame(
    X_train_encoded,
    columns=encoder.get_feature_names_out(X_train.select_dtypes(include=['object']).columns)
)
X_test_encoded_df = pd.DataFrame(
    X_test_encoded,
    columns=encoder.get_feature_names_out(X_test.select_dtypes(include=['object']).columns)
)

X_train_final = pd.concat([
    X_train.select_dtypes(exclude=['object']).reset_index(drop=True),
    X_train_encoded_df.reset_index(drop=True)
], axis=1)
X_test_final = pd.concat([
    X_test.select_dtypes(exclude=['object']).reset_index(drop=True),
    X_test_encoded_df.reset_index(drop=True)
], axis=1)

X_train_final.columns = X_train_final.columns.astype(str)
X_test_final.columns = X_test_final.columns.astype(str)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_final, y_train)

y_pred = model.predict(X_test_final)
#FIXED
# Add zero_division to handle cases with no predicted samples for a class
tprint(classification_report(y_test, y_pred, zero_division=0))
