```python
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "5050_split.csv")
data = pd.read_csv(raw_data_file)

# Splitting the data into training and test sets
X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data first to handle imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train) # MODIFIED: SMOTE applied before feature selection

# Perform feature selection on the resampled training data
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled) # MODIFIED: Feature selection fitted on resampled training data
X_test_selected = selector.transform(X_test) # MODIFIED: Test data transformed using selector fitted on resampled training data

# Standardize the data after feature selection, fitting only on the training data
scaler = StandardScaler()
X_train_final = scaler.fit_transform(X_train_selected) # MODIFIED: Scaler fitted on feature-selected, resampled training data
X_test_final = scaler.transform(X_test_selected) # MODIFIED: Test data transformed using scaler fitted on training data

# Training the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_final, y_train_resampled)

# Evaluating the model
y_pred = model.predict(X_test_final)
print(classification_report(y_test, y_pred))
```