# Summary of fixes:
# - Combined age and high cholesterol filters into a single expression for clarity.
# - Dropped rows with missing target or feature values to avoid errors during training.
# - Added stratify=y to train_test_split to preserve class distribution in train and test sets.
# - Simplified categorical encoding with pandas.get_dummies (drop_first=True to avoid multicollinearity).
# - Used DataFrame.align with join='left' to ensure train and test sets have the same feature columns.
# - Included both object and category dtypes when identifying categorical features.
# - Removed deprecated sparse_output parameter and OneHotEncoder in favor of pandas.get_dummies.
# - Split print statements to avoid relying on '\n' in string literals.

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, 'datasets', 'diabetes_indicator', '5050_split.csv')
data = pd.read_csv(raw_data_file)

# Drop rows with missing target or feature values
data = data.dropna(subset=['Diabetes_binary'])

# Filter unrealistic ages and ensure HighChol is a binary indicator (== 1)
data_filtered = data[(data['Age'] > 4) & (data['HighChol'] == 1)].copy()
data_filtered.dropna(inplace=True)

print('Raw data gender distribution:')
print(data_filtered['Sex'].value_counts(normalize=True).round(2))

X = data_filtered.drop('Diabetes_binary', axis=1)
y = data_filtered['Diabetes_binary']

# Stratified train-test split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print('Test set gender distribution:')
print(X_test['Sex'].value_counts(normalize=True).round(2))

# Identify categorical features (object or category dtypes)
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# One-hot encode categorical features using pandas.get_dummies
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Align train and test sets to have the same feature columns
X_train_final, X_test_final = X_train_encoded.align(
    X_test_encoded,
    join='left',
    axis=1,
    fill_value=0
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_final, y_train)

y_pred = model.predict(X_test_final)
print(classification_report(y_test, y_pred))
