"""
Summary of fixes:
- Added file existence check for raw data file.
- Added stratified train/test split to maintain class distribution.
- Added feature scaling using StandardScaler before model training.
- Set random_state in LogisticRegression for reproducibility.
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  #FIXED
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "binary_health_indicators.csv")
#FIXED
if not os.path.exists(raw_data_file):
    #FIXED
    raise FileNotFoundError(f"Raw data file not found: {raw_data_file}")
data = pd.read_csv(raw_data_file)

X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']
#FIXED
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#FIXED
scaler = StandardScaler()
#FIXED
X_train = scaler.fit_transform(X_train)
#FIXED
X_test = scaler.transform(X_test)

#FIXED
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
