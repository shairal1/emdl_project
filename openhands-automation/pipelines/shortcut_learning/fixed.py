# Summary of fixes:
# - Added __main__ guard to allow importing without execution.
# - Dropped missing values to prevent errors during modeling.
# - Performed train-test split before encoding to avoid data leakage.
# - Fitted LabelEncoders on training data and applied to test data.
# - Switched to RandomForestRegressor for continuous target variable.
# - Updated evaluation metrics to mean_squared_error and r2_score.
# - Set n_jobs=-1 for faster model training.
# - Improved formatting of evaluation metric outputs.

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
#FIXED
from sklearn.ensemble import RandomForestRegressor
#FIXED
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

#FIXED
def main():
    project_root = get_project_root()
    raw_data_file = os.path.join(project_root, "datasets", "alcohol", "Maths.csv")
    raw_data = pd.read_csv(raw_data_file)

    #FIXED
    # Drop rows with missing values
    raw_data = raw_data.dropna()

    X = raw_data.drop(columns=['G3'])
    y = raw_data['G3']

    #FIXED
    # Split data before encoding to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    label_encoders = {}
    for column in X_train.select_dtypes(include=['object']).columns:
        #FIXED
        label_encoders[column] = LabelEncoder()
        #FIXED
        X_train[column] = label_encoders[column].fit_transform(X_train[column])
        #FIXED
        X_test[column] = label_encoders[column].transform(X_test[column])

    #FIXED
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    #FIXED
    model.fit(X_train, y_train)

    #FIXED
    y_pred = model.predict(X_test)

    #FIXED
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    #FIXED
    print(f"R^2 Score: {r2_score(y_test, y_pred):.2f}")

#FIXED
if __name__ == "__main__":
    main()
