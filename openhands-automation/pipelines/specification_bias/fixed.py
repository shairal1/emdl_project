"""
Summary of fixes applied to the original pipeline:

- Handled missing values represented as '?' by using na_values parameter in pd.read_csv.
- Removed unnecessary sys.path manipulation and import of get_project_root from utils.
- Simplified dataset path resolution to be relative to the script location (__file__).
- Specified OneHotEncoder(sparse=False) to produce a dense array compatible with RandomForestClassifier.
- Mapped target labels (salary) to binary values (0 and 1) after stripping whitespace.
- Dropped any rows where target mapping resulted in NaN to ensure consistency.
- Added stratify=y to train_test_split to maintain class balance in splits.
- Set n_jobs=-1 and random_state=42 in RandomForestClassifier for reproducibility and performance.

"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Determine the directory where this script resides
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_file = os.path.join(script_dir, "datasets", "adult_data", "adult_data.csv")

# Load data, treating '?' as missing values
data = pd.read_csv(raw_data_file, na_values=['?'])

# Separate features and target
X = data.drop(columns=['salary']).copy()
# Strip whitespace and map salary to binary labels
y = data['salary'].str.strip().map({'<=50K': 0, '>50K': 1})

# Drop rows where target conversion failed (if any)
mask = y.notna()
X = X[mask]
y = y[mask]

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ]), categorical_cols)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Split data with stratification to preserve class ratios
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification report:")
print(classification_report(y_test, y_pred))
