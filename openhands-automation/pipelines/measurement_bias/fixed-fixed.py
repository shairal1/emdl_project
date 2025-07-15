# Summary of fixes:
# - Separated features and target before preprocessing to avoid transforming the target.
# - Identified numeric and categorical features on feature set X, excluding the target.
# - Applied imputation and scaling only to feature set X.
# - Added one-hot encoding for categorical features using pandas.get_dummies.
# - Added random_state to LogisticRegression for reproducibility.

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "student_data", "dataset.csv")
data = pd.read_csv(raw_data_file)

# Separate features and target
X = data.drop('Target', axis=1)  #FIXED
y = data['Target']  #FIXED

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=['number']).columns  #FIXED
categorical_features = X.select_dtypes(include=['object', 'category']).columns  #FIXED

# Impute missing values
numeric_imputer = SimpleImputer(strategy='mean')
X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])  #FIXED

categorical_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])  #FIXED

# Encode categorical features
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)  #FIXED

# Scale numeric features
scaler = StandardScaler()  # or MinMaxScaler() based on data needs  #FIXED
X[numeric_features] = scaler.fit_transform(X[numeric_features])  #FIXED

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(max_iter=1000, random_state=42)  #FIXED
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("Training set statistics:")
print(X_train.describe())
print("Test set statistics:")
print(X_test.describe())
