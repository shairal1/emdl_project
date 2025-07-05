```python
import os
import sys
import pandas as pd
import joblib # Added for model persistence
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score # Added StratifiedKFold and cross_val_score
from sklearn.preprocessing import StandardScaler # Added for feature scaling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from collections import Counter # Added to check class imbalance
import warnings # Added to suppress warnings for cleaner output

# Suppress all warnings for cleaner output in production-like scripts
warnings.filterwarnings('ignore')

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "binary_health_indicators.csv")
data = pd.read_csv(raw_data_file)

# --- Lack of data exploration & Imbalanced data handling (initial check) ---
# Display basic information about the dataset
print("--- Data Info ---")
data.info()
print("\n--- Data Description ---")
print(data.describe())
print("\n--- Missing Values ---")
print(data.isnull().sum())

# Check for class imbalance in the target variable
print("\n--- Target Variable Distribution ---")
target_distribution = data['Diabetes_binary'].value_counts()
print(target_distribution)
if target_distribution.min() / target_distribution.sum() < 0.2: # Example threshold for imbalance
    print(f"\nWarning: The target variable 'Diabetes_binary' is imbalanced. "
          f"Minority class ratio: {target_distribution.min() / target_distribution.sum():.2f}")

# Splitting the raw data into training and test sets
X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Added stratify=y to ensure the train/test split maintains the same proportion of target classes as the original dataset.

# --- No feature scaling ---
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrame for consistency if needed, though not strictly necessary for LogisticRegression
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


# --- Imbalanced data handling (via class weights) ---
# Training the model with class_weight='balanced' to handle potential class imbalance
# This automatically adjusts weights inversely proportional to class frequencies.
model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear') # Added class_weight='balanced' and specified solver for robustness

# Fit the model on the scaled training data
model.fit(X_train_scaled, y_train)

# --- Lack of cross-validation ---
# Perform cross-validation to get a more robust estimate of model performance
# Using StratifiedKFold to maintain class distribution in each fold
print("\n--- Cross-validation Performance (Accuracy) ---")
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Using 5 folds
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})") # Print mean and 2*std for confidence interval

# Evaluating the model on the held-out test set
print("\n--- Model Performance on Test Set ---")
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# --- Missing model persistence ---
# Define directory for saving models
model_dir = os.path.join(project_root, "models")
os.makedirs(model_dir, exist_ok=True) # Ensure the directory exists

# Define path for saving the model
model_path = os.path.join(model_dir, "logistic_regression_model.joblib")

# Save the trained model to disk
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")

# Optionally, you can load the model back to verify
# loaded_model = joblib.load(model_path)
# print(f"Model successfully loaded: {loaded_model}")
```