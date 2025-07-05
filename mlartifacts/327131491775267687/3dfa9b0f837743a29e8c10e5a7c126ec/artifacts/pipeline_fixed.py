```python
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score # Modified: Added roc_auc_score and f1_score
from sklearn.preprocessing import StandardScaler # Modified: Added StandardScaler

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "binary_health_indicators.csv")
data = pd.read_csv(raw_data_file)

# Checking class distribution in the raw data
print("Class distribution in raw data:\n", data['Income'].value_counts(normalize=True).round(2))

# Splitting the data into training and test sets
X = data.drop('Income', axis=1)
y = data['Income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modified: Added Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modified: Addressed Class Imbalance using class_weight='balanced'
model = LogisticRegression(max_iter=5000, class_weight='balanced')
model.fit(X_train_scaled, y_train) # Modified: Fit on scaled data

# Evaluating the model
y_pred = model.predict(X_test_scaled) # Modified: Predict on scaled data
print(classification_report(y_test, y_pred))

# Modified: Added more Evaluation Metrics
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Get probabilities for ROC AUC
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
```