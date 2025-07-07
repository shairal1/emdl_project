# Summary of fixes:
# - Added dataset file existence check before loading.
# - Used stratified splitting in train_test_split to maintain class balance.
# - Enabled parallel processing in RandomForestClassifier by setting n_jobs=-1.
# - Improved classification report printing for readability.
# - Formatted accuracy output to 4 decimal places.

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "binary_health_indicators.csv")
#FIXED
if not os.path.isfile(raw_data_file):
    raise FileNotFoundError(f"Dataset file not found: {raw_data_file}")
data = pd.read_csv(raw_data_file)

X = data.drop(columns=['Diabetes_binary'])
y = data['Diabetes_binary']

#FIXED
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#FIXED
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
#FIXED
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
#FIXED
print("Classification report:")
#FIXED
print(classification_report(y_test, y_pred))
