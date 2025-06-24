# fixed.py
# Summary of fixes:
# - Removed dependency on utils.get_project_root and dynamic sys.path modification; compute project root via os.path.
# - Handled missing values by dropping rows with NaNs in the target column.
# - Replaced LabelEncoder with pandas.get_dummies for one-hot encoding of categorical features to avoid implicit ordinality.
# - Switched from RandomForestClassifier to RandomForestRegressor for predicting the continuous G3 grade.
# - Used appropriate regression metrics (R^2 and RMSE) instead of classification metrics.
# - Improved print formatting for clarity and readability.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def main():
    # Compute project root as parent directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))

    # Load data
    raw_data_file = os.path.join(project_root, "datasets", "alcohol", "Maths.csv")
    df = pd.read_csv(raw_data_file)

    # Drop rows with missing target
    df = df.dropna(subset=["G3"])

    # Separate features and target
    X = df.drop(columns=["G3"])
    y = df["G3"]

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("Regression Performance on G3:")
    print(f"R^2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
