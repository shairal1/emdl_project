```python
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Modified: Changed from Classifier to Regressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # Modified: Imported regression metrics
from sklearn.preprocessing import LabelEncoder

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "alcohol", "Maths.csv")
raw_data = pd.read_csv(raw_data_file)

# Known dataset with high-correlation between G1, G2 and G3 features
# Modified: Removed 'G1' and 'G2' from features to prevent data leakage
X = raw_data.drop(columns=['G1', 'G2', 'G3'])
y = raw_data['G3']

# Encoding categorical variables
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a simple regressor
# Modified: Changed to RandomForestRegressor as G3 is a numerical grade
clf = RandomForestRegressor(random_state=42)
clf.fit(X_train, y_train)

# Prediction and evaluation
y_pred = clf.predict(X_test)
# Modified: Switched to regression evaluation metrics
print(f"R-squared (R2) Score: {r2_score(y_test, y_pred)}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred)}")
```