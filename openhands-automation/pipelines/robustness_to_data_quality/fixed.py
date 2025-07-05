"""
Summary of fixes:
- Handle missing values: replace '?' with NaN, impute numeric and categorical features.
- Prevent unseen category errors: set handle_unknown='ignore' in OneHotEncoder.
- Scale numeric features with StandardScaler for consistency.
- Ensure reproducibility: set random_state for classifier and np.random.seed for noise.
- Provide fallback for get_project_root import to avoid import errors.
- Encapsulate script execution under __main__ guard.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Attempt to import get_project_root; fallback to parent directory logic
try:
    from utils import get_project_root
except ImportError:
    def get_project_root():
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    # Reproducibility
    np.random.seed(42)

    # Locate dataset
    project_root = get_project_root()
    raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")

    # Load data, treating '?' as missing and trimming spaces
    data = pd.read_csv(raw_data_file, na_values='?', skipinitialspace=True)

    # Separate features and target
    X = data.drop(columns='salary')
    y = data['salary']

    # Encode target labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))

    # Identify feature types
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['number']).columns.tolist()

    # Preprocessing pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Full pipeline with Random Forest classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate before noise
    y_pred = pipeline.predict(X_test)
    acc_before = accuracy_score(y_test, y_pred)

    # Inject Gaussian noise into numeric features
    X_test_noisy = X_test.copy()
    num_cols = X_test_noisy.select_dtypes(include=['number']).columns
    noise = np.random.normal(loc=0, scale=15, size=X_test_noisy[num_cols].shape)
    X_test_noisy[num_cols] += noise

    # Evaluate after noise
    y_pred_noisy = pipeline.predict(X_test_noisy)
    acc_after = accuracy_score(y_test, y_pred_noisy)

    # Report results
    print(f"Accuracy before noise: {acc_before:.4f}")
    print(f"Accuracy after noise: {acc_after:.4f}")


if __name__ == '__main__':
    main()
