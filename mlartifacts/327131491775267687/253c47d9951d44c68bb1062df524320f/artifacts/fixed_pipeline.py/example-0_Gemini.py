import sys
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline # Import Pipeline from imblearn for ML pipelines

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root
import pandas as pd # Moved here to be with other imports

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "5050_split.csv")
data = pd.read_csv(raw_data_file)

# [Lack of Data Splitting for Final Evaluation]: Split data into training and a separate test set
# The test set is held out and not used during cross-validation, providing an unbiased evaluation.
X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# [Data Leakage] & [Unnecessary Feature Selection]:
# Encapsulate preprocessing steps (undersampling, feature selection) and the classifier into a Pipeline.
# This ensures that undersampling and feature selection are performed *within* each cross-validation fold
# and only on the training data of that fold, preventing data leakage.
# Feature selection is now applied per fold after undersampling, addressing the optimality concern by
# ensuring it operates on the balanced data for each fold.
pipeline = Pipeline([
    ('undersampler', RandomUnderSampler(random_state=42)),
    ('feature_selection', SelectKBest(f_classif, k=10)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Cross-validation
# The pipeline handles all steps for each fold automatically.
scores = cross_val_score(pipeline, X_train, y_train, cv=5)

print("Cross-validation scores:", scores)
print("Mean CV accuracy:", scores.mean())

# [Lack of Data Splitting for Final Evaluation]:
# Train the final model on the entire training set (X_train, y_train)
# and evaluate its performance on the unseen test set (X_test, y_test).
pipeline.fit(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)
print("Test set accuracy:", test_accuracy)