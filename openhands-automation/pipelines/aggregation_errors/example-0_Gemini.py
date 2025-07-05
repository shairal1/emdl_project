import sys
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler # Modified: Removed Normalizer, added StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score # Modified: Added cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import numpy as np # Modified: Added numpy for mean/std of cross-validation scores

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

# [Incorrect Text Normalization]
# Problem: The original code included `data['occupation'] = data['occupation'].str.replace('-', ' ')`,
# which was an example of incorrect or oversimplified text aggregation/normalization.
# Solution: Removed the `str.replace` line. Basic lowercasing is kept as a simple, non-problematic step.
# More advanced text normalization (stemming, lemmatization, stop word removal) would typically
# involve libraries like NLTK or SpaCy and more complex pipeline components, which are beyond the scope
# of simple string operations for this dataset.
data['occupation'] = data['occupation'].str.lower()

# [Incorrect Spatial Aggregation]
# Problem: The original code aggregated all 'native-country' values into a single 'North America' category,
# leading to a significant loss of information.
# Solution: Removed the line performing this aggregation. The OneHotEncoder will now correctly handle
# 'native-country' as a distinct categorical feature.
# Removed: data['native-country'] = data['native-country'].apply(lambda x: 'North America')

# Splitting data
# Added random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(data.drop('salary', axis=1), data['salary'], test_size=0.2, random_state=42)

# Defining preprocessing for numeric and categorical features
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    # [Missing Feature Scaling]
    # Problem: Normalizer was used, which scales each sample to unit norm. This is often not
    # the optimal choice for general feature scaling, especially when compared to StandardScaler or MinMaxScaler.
    # It was also applied before OneHotEncoding, but scaling is typically only for numeric features.
    # Solution: Replaced Normalizer with StandardScaler, which is a more common and generally
    # beneficial scaler that standardizes features by removing the mean and scaling to unit variance.
    ('scaler', StandardScaler()) # Modified: Changed from Normalizer() to StandardScaler()
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Combining preprocessing with the classifier
# Added random_state to RandomForestClassifier for reproducibility
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# [No Cross-Validation]
# Problem: The original model was evaluated using a single train-test split, which provides a
# less robust estimate of generalization performance.
# Solution: Implemented 5-fold cross-validation on the training data using `cross_val_score`
# to get a more reliable performance estimate.
print("Performing cross-validation on the training data...")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print(f"Cross-validation accuracy: {np.mean(cv_scores):.2f} +/- {np.std(cv_scores):.2f}")

# Fitting the final model on the entire training data and evaluating on the held-out test set
# This step is performed after cross-validation to get a final model trained on all available
# training data and to provide a detailed classification report on unseen test data.
print("\nFitting final model and evaluating on held-out test set...")
pipeline.fit(X_train, y_train)

# Evaluating the model on the held-out test set
score = pipeline.score(X_test, y_test)
print(f"Model accuracy on test set: {score:.2f}")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# [Potential Data Leakage]
# The initial train-test split and the use of sklearn Pipelines ensure that
# imputation and scaling are fitted only on the training data (X_train) and then applied
# consistently to both training and test data. The manual modifications to 'occupation'
# were simple string operations not dependent on dataset statistics, thus not introducing
# leakage. Removing the problematic 'native-country' aggregation further cleans the data
# pre-processing steps, adhering to best practices to prevent common forms of data leakage.