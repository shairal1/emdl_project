# Summary of fixes:
# - Removed unnecessary sys.path manipulation; import get_project_root directly.
# - Wrapped execution in a main() function with if __name__ == "__main__" guard.
# - Added error handling for missing data file (FileNotFoundError).
# - Encoded target variable 'Income' with LabelEncoder for numeric compatibility.
# - Preprocessed features: scaled numeric columns with StandardScaler and one-hot encoded categorical columns with OneHotEncoder.
# - Combined preprocessing and model into a sklearn Pipeline for clarity and robustness.
# - Set random_state in train_test_split and LogisticRegression for reproducibility.
# - Enhanced classification report to display original class names.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils import get_project_root

def main():
    # Determine project root and data path
    project_root = get_project_root()
    raw_data_file = os.path.join(
        project_root,
        "datasets",
        "diabetes_indicator",
        "binary_health_indicators.csv"
    )

    # Load data with error handling
    try:
        data = pd.read_csv(raw_data_file)
    except FileNotFoundError:
        print(f"Error: Data file not found at {raw_data_file}")
        return

    # Display class distribution
    print("Class distribution in raw data:\n",
          data['Income'].value_counts(normalize=True).round(2))

    # Separate features and target
    X = data.drop('Income', axis=1)
    y = data['Income']

    # Encode the target variable as numeric
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42
    )

    # Identify numeric and categorical feature columns
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
        ]
    )

    # Complete modeling pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=5000, random_state=42))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))


if __name__ == "__main__":
    main()
