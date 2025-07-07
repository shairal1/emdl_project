# Summary of fixes:
# 1. Added stratify=y to train_test_split to maintain target distribution.
# 2. Scaled features before oversampling and feature selection to ensure proper normalization.
# 3. Applied SMOTE on scaled training data to generate synthetic samples on normalized features.
# 4. Performed feature selection after SMOTE to select features based on balanced data.
# 5. Specified random_state in LogisticRegression for reproducibility.

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "5050_split.csv")
data = pd.read_csv(raw_data_file)

X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']
#FIXED: Added stratify parameter to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#FIXED: Scale features before SMOTE and feature selection
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#FIXED: Applied SMOTE on scaled training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

#FIXED: Performed feature selection after SMOTE
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = selector.transform(X_test_scaled)

#FIXED: Set random_state for reproducibility in LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_selected, y_train_resampled)


y_pred = model.predict(X_test_selected)
print(classification_report(y_test, y_pred))