# Summary of fixes:
# 1. Handle missing values represented by '?', replaced with NaN and dropped incomplete rows.
# 2. Drop unnecessary columns in-place.
# 3. Encode categorical features using one-hot encoding instead of LabelEncoder.
# 4. Encode target variable separately using LabelEncoder.
# 5. Use stratified split to maintain class distribution.
# 6. Improve classification report printing with target names.

import os
import sys
import pandas as pd
import numpy as np  #FIXED
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

data.replace(' ?', np.nan, inplace=True)  #FIXED
data.dropna(inplace=True)  #FIXED

data.drop(columns=['education', 'occupation'], inplace=True)  #FIXED

X_raw = data.drop(columns=['salary'])  #FIXED
y_raw = data['salary']  #FIXED

le_target = LabelEncoder()  #FIXED
y = le_target.fit_transform(y_raw)  #FIXED

X = pd.get_dummies(X_raw, drop_first=True)  #FIXED

X_train, X_test, y_train, y_test = train_test_split(  #FIXED
    X, y, test_size=0.2, random_state=42, stratify=y  #FIXED
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')  #FIXED
print('Classification report:')  #FIXED
print(classification_report(y_test, y_pred, target_names=le_target.classes_))  #FIXED