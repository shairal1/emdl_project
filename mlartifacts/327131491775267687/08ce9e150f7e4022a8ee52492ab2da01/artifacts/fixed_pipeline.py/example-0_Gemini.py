import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler # MODIFICATION: Added StandardScaler
from sklearn.compose import ColumnTransformer # MODIFICATION: Added ColumnTransformer
from sklearn.pipeline import Pipeline # MODIFICATION: Added Pipeline

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "5050_split.csv")
data = pd.read_csv(raw_data_file)

# Checking the distribution of protected feature classes in the raw data
print("Raw data gender distribution:\n", data['Sex'].value_counts(normalize=True).round(2))

# MODIFICATION: [Data Filtering Bias] - Removed data filtering before splitting
# The original code filtered the entire dataset based on 'Age' and 'HighChol'
# before splitting, which can make the test set unrepresentative of the original
# population and introduce bias. The filtering lines have been removed.
# If such filtering is a strict domain requirement, it should be carefully
# applied or justified within the pipeline after the split.

# Splitting the data into training and test sets from the original, unfiltered dataset
X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']
# MODIFICATION: [Potential Data Imbalance] - Added stratify to train_test_split
# Stratification ensures that the proportion of the target variable ('Diabetes_binary')
# is maintained in both the training and test sets, helping to address potential imbalance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Checking the distribution of protected feature classes in the test set (after correct split)
print("Test set gender distribution:\n", X_test['Sex'].value_counts(normalize=True).round(2))

# Identify numerical and categorical features
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# MODIFICATION: [Lack of Feature Scaling] & [Inconsistent Data Transformation]
# Numerical features were not scaled, which can negatively impact Logistic Regression.
# The previous manual handling of transformations could lead to inconsistencies.
# Using ColumnTransformer within a Pipeline ensures consistent scaling of numerical
# features and one-hot encoding of categorical features, and automatically handles
# column alignment for both training and test sets.
preprocessor = ColumnTransformer(
    transformers=[
        # Apply StandardScaler to numerical features for scaling
        ('num', StandardScaler(), numerical_features),
        # Apply OneHotEncoder to categorical features with 'handle_unknown' and 'drop'
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features)
    ],
    remainder='passthrough' # Keep any other columns not explicitly transformed
)

# MODIFICATION: [Potential Data Imbalance] - Added class_weight='balanced' to LogisticRegression
# To address potential class imbalance in the target variable, 'class_weight='balanced''
# is used. This automatically adjusts weights inversely proportional to class frequencies,
# giving more importance to the minority class.
# MODIFICATION: Using Pipeline to combine preprocessor and model for robustness
# This streamlines the workflow, ensuring pre-processing steps are applied consistently
# to new data (e.g., test set or future predictions).
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Training the model
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))