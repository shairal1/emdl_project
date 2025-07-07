"""
Summary of fixes:
- Added stratified train-test split to maintain class distribution (stratify=y).
- Handled missing values by specifying na_values and dropping rows with missing data.
- Standardized features using StandardScaler before model fitting for numerical stability.
- Wrapped execution code in a main() function with a __name__ == "__main__" guard.
- Added error handling for missing data file.
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#FIXED
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

#FIXED

def load_data(file_path):
    #FIXED
    if not os.path.exists(file_path):
        #FIXED
        raise FileNotFoundError(f"Data file not found: {file_path}")
    #FIXED
    data = pd.read_csv(file_path, na_values=["?", "NA", ""])
    #FIXED
    missing = data.isnull().sum().sum()
    #FIXED
    if missing > 0:
        #FIXED
        print(f"Found {missing} missing values. Dropping rows.")
        #FIXED
        data = data.dropna()
    #FIXED
    return data

#FIXED

def main():
    project_root = get_project_root()
    raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "binary_health_indicators.csv")
    #FIXED
    data = load_data(raw_data_file)

    print("Class distribution in raw data:\n", data['Income'].value_counts(normalize=True).round(2))

    X = data.drop('Income', axis=1)
    y = data['Income']
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

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

#FIXED
if __name__ == "__main__":
    #FIXED
    main()
