```python
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score # Modified: Added StratifiedKFold, cross_val_score for cross-validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler # Modified: Added StandardScaler for feature scaling
from sklearn.pipeline import Pipeline # Modified: Added Pipeline to chain preprocessing and model
import joblib # Modified: Added joblib for model persistence

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

# Modified: Lack of data exploration (EDA)
# Perform basic exploratory data analysis to understand data characteristics
print("--- Data Exploration ---")
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Information:")
data.info()
print("\nDescriptive Statistics:")
print(data.describe())
print("\nTarget variable (Diabetes_binary) distribution:")
target_distribution = data['Diabetes_binary'].value_counts(normalize=True)
print(target_distribution)


# Splitting dataset into features and target
X = data.drop(columns=['Diabetes_binary'])
y = data['Diabetes_binary']

# Modified: Splitting data into training and test sets with stratification
# Stratify ensures that the proportion of target classes is roughly the same in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Modified: Imbalanced data handling
# Check if the target is significantly imbalanced and apply class_weight if necessary.
imbalance_threshold = 0.2
model_params = {'random_state': 42, 'n_estimators': 100}
if target_distribution.min() < imbalance_threshold:
    print(f"\n--- Imbalanced Data Handling ---")
    print(f"Detected class imbalance: Minority class is {target_distribution.min():.2%}. Applying class_weight='balanced'.")
    # For RandomForest, class_weight='balanced' adjusts weights inversely proportional to class frequencies.
    model_params['class_weight'] = 'balanced'
else:
    print(f"\n--- Imbalanced Data Check ---")
    print(f"Class distribution appears balanced enough ({target_distribution.min():.2%}). Not applying class_weight.")


# Modified: Feature Scaling and Model Pipeline
# Create a pipeline to chain StandardScaler (for feature scaling) and RandomForestClassifier.
# StandardScaler transforms features to have zero mean and unit variance, beneficial for many models.
print("\n--- Feature Scaling and Model Pipeline ---")
pipeline = Pipeline([
    ('scaler', StandardScaler()), # Step 1: Feature Scaling
    ('classifier', RandomForestClassifier(**model_params)) # Step 2: RandomForest Model
])
print("Pipeline created: StandardScaler -> RandomForestClassifier")


# Modified: Missing cross-validation
# Perform cross-validation for a more robust evaluation of the model's generalization performance.
print("\n--- Performing Cross-Validation ---")
# Use StratifiedKFold to maintain the proportion of target variable in each fold during CV.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Using 5 folds as a common practice

cv_results = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"Cross-validation Accuracy Scores: {cv_results}")
print(f"Mean CV Accuracy: {cv_results.mean():.4f} (+/- {cv_results.std():.4f})")


# Training the final model on the training data (using the pipeline)
print("\n--- Training Final Model ---")
pipeline.fit(X_train, y_train)
print("Model training complete.")


# Prediction and evaluation on the held-out test set
print("\n--- Evaluation on Test Set ---")
y_pred = pipeline.predict(X_test) # Predict using the fitted pipeline
print(f"Accuracy on Test Set: {accuracy_score(y_test, y_pred):.4f}")
print(f"Classification Report on Test Set:\n{classification_report(y_test, y_pred)}")


# Modified: Lack of model persistence
# Save the trained model (pipeline) to disk for future use without retraining.
model_dir = os.path.join(project_root, "models")
os.makedirs(model_dir, exist_ok=True) # Ensure the directory exists
model_filename = os.path.join(model_dir, "diabetes_model.joblib")

print(f"\n--- Model Persistence ---")
print(f"Saving trained model to: {model_filename}")
joblib.dump(pipeline, model_filename)
print("Model saved successfully.")
```