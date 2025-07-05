# Summary of fixes:
# 1. Removed unnecessary sys.path manipulation and get_project_root usage; using relative dataset path.
# 2. Removed the erroneous replacement of all 'native-country' values with a constant.
# 3. Applied consistent text cleaning to 'occupation' and 'native-country' (lowercasing, hyphen replacement with regex=False).
# 4. Replaced Normalizer with StandardScaler for numeric features.
# 5. Added random_state and stratify to train_test_split, and random_state to RandomForestClassifier for reproducibility.
# 6. Specified regex=False in str.replace to avoid FutureWarning.

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

# Load dataset (assumes current working directory is project root)
raw_data_file = os.path.join("datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

# Text cleaning for categorical columns
for col in ['occupation', 'native-country']:
    if col in data.columns and data[col].dtype == object:
        data[col] = data[col].str.lower()
        data[col] = data[col].str.replace('-', ' ', regex=False)

# Separate features and target
X = data.drop('salary', axis=1)
y = data['salary']

# Split data with stratification and fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Identify feature types
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing numeric values with median
    ('scaler', StandardScaler())                     # Standardize numeric features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Fill missing categorical
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))       # One-hot encode categories
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Build the full pipeline
two_step_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
two_step_pipeline.fit(X_train, y_train)

# Evaluate the model
accuracy = two_step_pipeline.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Detailed classification report
y_pred = two_step_pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
