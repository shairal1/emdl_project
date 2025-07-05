```python
import pandas as pd
import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate # Modified: Added StratifiedKFold, cross_validate
import joblib # Modified: Added for model persistence

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
# Modified: [Data leakage] Note on data license verification.
# Note: The license for 'adult_data.csv' should be verified for permissible use in this context.
data = pd.read_csv(raw_data_file)

# Modified: [Missing feature engineering]
# Feature Engineering
# Create age bins to capture non-linear relationships with age
data['age_bins'] = pd.cut(data['age'], bins=[0, 25, 45, 65, 100],
                           labels=['Young', 'Adult', 'Senior', 'Elderly'], right=False,
                           include_lowest=True) # Ensure all ages are covered

# Create a binary feature for native country (USA vs. Non-USA)
data['is_usa'] = (data['native-country'] == 'United-States').astype(int)

# Defining column names - Modified: Included new engineered features
numeric_columns = ['age', 'hours-per-week']
# Added 'age_bins' as categorical, 'is_usa' as categorical (OneHotEncoder can handle 0/1 integers)
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'native-country', 'age_bins', 'is_usa']

# Defining the target variable
target = 'salary'

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combining preprocessors into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Splitting the data into features and target
X = data[numeric_columns + categorical_columns]
y = data[target]

# Encoding the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Modified: Removed train_test_split as cross-validation will be used for robust evaluation.
# Creating the model pipeline - Modified: [Unbalanced data handling], [Hyperparameter optimization]
# Added class_weight='balanced' to LogisticRegression to automatically adjust weights
# inversely proportional to class frequencies, addressing class imbalance.
# A robust solver ('liblinear') and random_state are also added for consistency,
# which are general improvements rather than extensive hyperparameter tuning.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'))
])

# Modified: [Lack of cross-validation] - Implementing cross-validation
# Define stratified k-fold cross-validation strategy for robust model evaluation,
# ensuring each fold has a similar proportion of target classes.
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation to evaluate model performance across multiple folds
print("Performing cross-validation...")
cv_results = cross_validate(model, X, y_encoded, cv=cv_strategy,
                            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                            return_train_score=False) # Only test scores are needed for evaluation

# Print cross-validation results summary (mean and standard deviation)
print("\nCross-validation results (mean +/- std):")
for metric in cv_results.keys():
    if metric.startswith('test_'):
        print(f"{metric.replace('test_', '').capitalize()}: {cv_results[metric].mean():.4f} (+/- {cv_results[metric].std():.4f})")

# Modified: [No model persistence]
# After evaluating performance with cross-validation, train the final model on the entire dataset
# before saving for deployment. This leverages all available data for the final model.
print("\nTraining final model on the full dataset for deployment...")
model.fit(X, y_encoded)

# Save the trained model to disk for later use
model_save_path = os.path.join(project_root, "models", "logistic_regression_model.pkl")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True) # Ensure the directory exists
joblib.dump(model, model_save_path)
print(f"Model successfully saved to: {model_save_path}")

# Removed the original classification_report print, as model evaluation is now handled by cross_validate.
```