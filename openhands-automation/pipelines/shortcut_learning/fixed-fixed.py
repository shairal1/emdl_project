"""
Summary of fixes:
- Removed unused import: resample
- Added import for compute_sample_weight to compute sample weights properly (#FIXED)
- Ensured reproducibility by setting random seed before augmentation (#FIXED)
- Scaled augmented training data using existing scaler before fitting model (#FIXED)
- Used separate classifier instances for each experiment to avoid unintended state carry-over (#FIXED)
- Computed sample weights with inverse class frequency using compute_sample_weight('balanced', ...) (#FIXED)
- Computed permutation importance on test set for unbiased feature importance (#FIXED)
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight #FIXED

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import get_project_root

project_root = get_project_root()

# Load data
raw_data_file = os.path.join(project_root, "datasets", "alcohol", "Maths.csv")
raw_data = pd.read_csv(raw_data_file)

# Split features and target
X = raw_data.drop(columns=['G3'])
y = raw_data['G3']

# Encode categorical features
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline model
clf_base = RandomForestClassifier(random_state=42) #FIXED
clf_base.fit(X_train_scaled, y_train)
perm_importance_base = permutation_importance(
    clf_base, X_test_scaled, y_test, n_repeats=10, random_state=42
) #FIXED
feature_importances_base = perm_importance_base.importances_mean
print(f"Baseline feature importances: {feature_importances_base}")

# Data augmentation
test
np.random.seed(42) #FIXED
X_train_aug = X_train.copy()
X_train_aug['G1'] += np.random.normal(0, 0.1, size=X_train.shape[0]) #FIXED
X_train_aug_scaled = scaler.transform(X_train_aug) #FIXED

clf_aug = RandomForestClassifier(random_state=42) #FIXED
clf_aug.fit(X_train_aug_scaled, y_train)
y_pred_aug = clf_aug.predict(X_test_scaled)
print(f"Accuracy after augmentation: {accuracy_score(y_test, y_pred_aug)}")

# Sample weighting
sample_weights = compute_sample_weight('balanced', y_train) #FIXED
clf_weighted = RandomForestClassifier(random_state=42) #FIXED
clf_weighted.fit(X_train_scaled, y_train, sample_weight=sample_weights)
y_pred_wt = clf_weighted.predict(X_test_scaled)
print(f"Accuracy with sample weighting: {accuracy_score(y_test, y_pred_wt)}")

perm_importance_wt = permutation_importance(
    clf_weighted, X_test_scaled, y_test, n_repeats=10, random_state=42
) #FIXED
feature_importances_wt = perm_importance_wt.importances_mean
print(f"Feature importances after augmentation and weighting: {feature_importances_wt}")
