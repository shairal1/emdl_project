"""
Summary of fixes:
1. Added stratify parameter to train_test_split to preserve class distribution.
2. Removed unnecessary sys.path manipulation for cleaner imports.
3. Moved scaling before SMOTE to ensure SMOTE operates on standardized features.
4. Added random_state to LogisticRegression for reproducibility.
5. Added handling for potential missing values in the dataset.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from utils import get_project_root

def main():
    # Locate the raw data file
    project_root = get_project_root()
    raw_data_file = os.path.join(
        project_root, "datasets", "diabetes_indicator", "5050_split.csv"
    )
    data = pd.read_csv(raw_data_file)

    # Handle missing values if any
    if data.isnull().values.any():
        data = data.dropna()

    # Separate features and target
    X = data.drop("Diabetes_binary", axis=1)
    y = data["Diabetes_binary"]

    # Split with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Select top-k features
    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Standardize features before oversampling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # Oversample the minority class
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled, y_train
    )

    # Train the classifier
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate on the test set
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
