# Summary of fixes:
# - Handled missing values encoded as '?' by replacing with NaN and dropping incomplete rows.
# - Switched OneHotEncoder parameter 'sparse_output' to 'sparse' for broader compatibility.
# - Scaled numerical features with StandardScaler.
# - Separated categorical and numerical columns explicitly before transformation.
# - Adjusted regularization strength C from 1e-4 to 1.0 to reduce underfitting.

import os
import sys
import pandas as pd
import numpy as np  #FIXED: Added numpy import for array operations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler  #FIXED: Import StandardScaler and use sparse for compatibility

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

# Handle missing values encoded as '?' by replacing with NaN and dropping rows with missing values
data.replace(' ?', pd.NA, inplace=True)  #FIXED
data.dropna(inplace=True)  #FIXED

X = data.drop('salary', axis=1)
y = data['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize encoder and scaler
encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')  #FIXED
scaler = StandardScaler()  #FIXED

# Identify categorical and numerical columns
cat_cols = X_train.select_dtypes(include=['object']).columns  #FIXED
num_cols = X_train.select_dtypes(exclude=['object']).columns  #FIXED

# Transform features
X_train_cat = encoder.fit_transform(X_train[cat_cols])  #FIXED
X_test_cat = encoder.transform(X_test[cat_cols])  #FIXED
X_train_num = scaler.fit_transform(X_train[num_cols])  #FIXED
X_test_num = scaler.transform(X_test[num_cols])  #FIXED

# Combine numeric and categorical features into final DataFrames
X_train_final = pd.DataFrame(
    np.hstack([X_train_num, X_train_cat]),
    columns=list(num_cols) + list(encoder.get_feature_names_out(cat_cols))
)  #FIXED
X_test_final = pd.DataFrame(
    np.hstack([X_test_num, X_test_cat]),
    columns=list(num_cols) + list(encoder.get_feature_names_out(cat_cols))
)  #FIXED

# Adjust regularization strength to default to reduce underfitting
model = LogisticRegression(max_iter=1000, C=1.0)  #FIXED
model.fit(X_train_final, y_train)

y_pred = model.predict(X_test_final)
print(classification_report(y_test, y_pred))
