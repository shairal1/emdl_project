import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# MODIFICATION: Import AIF360 components for bias mitigation and fairness metrics
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

# MODIFICATION: Acknowledge and define sensitive attributes for bias mitigation and fairness evaluation
# The 'adult' dataset is known to be biased. 'sex' and 'race' are identified as sensitive attributes.
# 'Male' and 'White' are typically considered privileged groups in this dataset for salary prediction.
sensitive_attribute_names = ['sex', 'race']
label_name = 'salary'
favorable_label = '>50K' # The label considered as the 'favorable' outcome (higher salary)

# Define privileged and unprivileged groups for Reweighing (focused on 'sex' for simplicity)
# Reweighing will adjust sample weights based on these definitions to reduce bias in training.
privileged_groups_reweigh = [{'sex': 'Male'}]
unprivileged_groups_reweigh = [{'sex': 'Female'}]

# Define privileged and unprivileged groups for fairness metrics (for both 'sex' and 'race' independently)
sex_privileged_groups = [{'sex': 'Male'}]
sex_unprivileged_groups = [{'sex': 'Female'}]

race_privileged_groups = [{'race': 'White'}]
# Include all other racial groups as unprivileged for comprehensive evaluation
race_unprivileged_groups = [
    {'race': 'Black'},
    {'race': 'Asian-Pac-Islander'},
    {'race': 'Amer-Indian-Eskimo'},
    {'race': 'Other'}
]

# Using known biased dataset
X = data.drop(label_name, axis=1)
y = data[label_name]

# MODIFICATION: Clean sensitive attributes from leading/trailing spaces
# This ensures consistency for AIF360's group identification and OneHotEncoder.
for col in sensitive_attribute_names:
    if col in X.columns and X[col].dtype == 'object':
        X[col] = X[col].str.strip()

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODIFICATION: Convert target labels to binary (0/1) for AIF360 StandardDataset compatibility
# AIF360 algorithms and metrics typically expect binary integer labels (0 or 1).
y_train_binary = y_train.apply(lambda x: 1 if x == favorable_label else 0)
y_test_binary = y_test.apply(lambda x: 1 if x == favorable_label else 0)

# MODIFICATION: Create AIF360 StandardDataset for training data
# This dataset format is required by AIF360 algorithms like Reweighing.
# Ensure sensitive attributes are included in the DataFrame passed to StandardDataset.
train_df_for_aif = pd.concat([X_train.reset_index(drop=True), y_train_binary.rename(label_name).reset_index(drop=True)], axis=1)

train_dataset = StandardDataset(
    train_df_for_aif,
    label_name=label_name,
    favorable_classes=[1], # 1 represents the favorable outcome (e.g., salary >50K)
    protected_attribute_names=sensitive_attribute_names,
    privileged_groups=privileged_groups_reweigh,
    unprivileged_groups=unprivileged_groups_reweigh
)

# MODIFICATION: Apply Reweighing bias mitigation technique to training data
# Reweighing adjusts the sample weights to mitigate bias by equalizing representation of privileged
# and unprivileged groups in the training data, with respect to the favorable outcome.
RW = Reweighing(
    unprivileged_groups=unprivileged_groups_reweigh,
    privileged_groups=privileged_groups_reweigh
)
train_dataset_reweighed = RW.fit_transform(train_dataset)
sample_weights_train = train_dataset_reweighed.instance_weights

# Identifying categorical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocessing the data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough' # Leave the rest of the columns untouched
)

# Creating the pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # MODIFICATION: Added solver='liblinear' to LogisticRegression to suppress a common warning.
    # Increased max_iter for convergence, which was already in original.
    ('classifier', LogisticRegression(max_iter=1000, solver='liblinear'))
])

# Training the model with preprocessing, passing the sample weights from Reweighing
# MODIFICATION: Pass generated sample weights to the classifier's fit method.
# This makes the Logistic Regression model more sensitive to the reweighted samples.
model_pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights_train)

# Evaluating the model on test set using standard classification metrics
y_pred = model_pipeline.predict(X_test)
print("--- Standard Classification Report (before fairness considerations) ---")
# Using y_test (original labels) for classification_report as it expects the same format as y_pred
print(classification_report(y_test, y_pred, zero_division=0))

# MODIFICATION: Evaluate fairness metrics using AIF360
print("\n--- Fairness Metrics (after bias mitigation with Reweighing) ---")

# Create AIF360 StandardDataset for the test set (with original true labels)
# This dataset is used as the reference for true outcomes and sensitive attributes.
test_df_for_aif = pd.concat([X_test.reset_index(drop=True), y_test_binary.rename(label_name).reset_index(drop=True)], axis=1)

test_dataset = StandardDataset(
    test_df_for_aif,
    label_name=label_name,
    favorable_classes=[1],
    protected_attribute_names=sensitive_attribute_names,
    # Define all relevant groups here for AIF360 dataset creation, even if metrics are evaluated separately.
    privileged_groups=sex_privileged_groups + race_privileged_groups,
    unprivileged_groups=sex_unprivileged_groups + race_unprivileged_groups
)

# Create a dataset with predicted labels for fairness metric calculation
# AIF360 metrics require predicted labels in a StandardDataset format as well.
test_dataset_pred = test_dataset.copy()
# Convert y_pred to binary 0/1 for AIF360 consistency, as y_pred contains original string labels.
y_pred_binary = pd.Series(y_pred).apply(lambda x: 1 if x == favorable_label else 0).values
test_dataset_pred.labels = y_pred_binary.reshape(-1, 1) # Reshape to (n_samples, 1) for AIF360

# Evaluate fairness for 'sex' as the protected attribute
print("\n--- Fairness Metrics for 'sex' ---")
metric_sex = ClassificationMetric(
    test_dataset,
    test_dataset_pred,
    unprivileged_groups=sex_unprivileged_groups,
    privileged_groups=sex_privileged_groups
)
print(f"Disparate Impact (should be close to 1 for fairness): {metric_sex.disparate_impact():.4f}")
print(f"Average Odds Difference (should be close to 0 for fairness): {metric_sex.average_odds_difference():.4f}")
print(f"Equal Opportunity Difference (should be close to 0 for fairness): {metric_sex.equal_opportunity_difference():.4f}")

# Evaluate fairness for 'race' as the protected attribute
print("\n--- Fairness Metrics for 'race' ---")
metric_race = ClassificationMetric(
    test_dataset,
    test_dataset_pred,
    unprivileged_groups=race_unprivileged_groups,
    privileged_groups=race_privileged_groups
)
print(f"Disparate Impact (should be close to 1 for fairness): {metric_race.disparate_impact():.4f}")
print(f"Average Odds Difference (should be close to 0 for fairness): {metric_race.average_odds_difference():.4f}")
print(f"Equal Opportunity Difference (should be close to 0 for fairness): {metric_race.equal_opportunity_difference():.4f}")