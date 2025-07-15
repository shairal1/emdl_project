# Summary of fixes:
# - Integrated SMOTE into pipeline using imblearn.pipeline.Pipeline
# - Removed manual SMOTE resampling outside the pipeline
# - Reordered pipeline steps: scaling, SMOTE, feature selection, classifier
# - Added random_state to LogisticRegression for reproducibility

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  #FIXED
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaling', StandardScaler()),  # Apply scaling
    ('smote', SMOTE(random_state=42)),  #FIXED
    ('feature_selection', SelectKBest(f_classif, k=10)),  # Applying feature selection
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))  #FIXED
])

pipeline.fit(X_train, y_train)  #FIXED

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
