import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "titanic", "data.csv")
data = pd.read_csv(raw_data_file)

# --- Start of Modifications ---

# [Insufficient Feature Engineering] & [Inadequate Missing Value Handling] for 'Cabin', 'Name'
# 1. Engineer 'Has_Cabin' feature from 'Cabin'
data['Has_Cabin'] = data['Cabin'].notna().astype(int)

# 2. Extract 'Title' from 'Name'
data['Title'] = data['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
# Group less common titles to reduce cardinality
common_titles = ['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev']
data['Title'] = data['Title'].apply(lambda x: x if x in common_titles else 'Rare')

# Define features (X) and target (y)
# Drop original 'Name', 'Ticket', 'Cabin', 'PassengerId' as they are either engineered or not directly useful
X = data.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
y = data['Survived']

# [Potential Data Imbalance] - Check class distribution
# print("Target variable distribution:\n", y.value_counts(normalize=True))
# While not explicitly handled with techniques like SMOTE, stratifying the split helps maintain distribution.

# [Data Leakage] & [Lack of Feature Scaling] & [Inadequate Missing Value Handling]
# Create preprocessing pipelines for numerical and categorical features
# Numerical features to be imputed and scaled
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Impute missing 'Age' and 'Fare' (if any)
    ('scaler', StandardScaler())                 # Scale numerical features
])

# Categorical features to be imputed (if needed) and one-hot encoded
categorical_features = ['Sex', 'Embarked', 'Pclass', 'Has_Cabin', 'Title']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing 'Embarked'
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # One-hot encode categorical features
])

# Create a preprocessor using ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop columns not specified in numerical_features or categorical_features
)

# Encoding labels (already binary, but good practice to ensure consistency if not 0/1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting data (with stratification for [Potential Data Imbalance])
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Create the full pipeline including preprocessing and the model
# [Overfitting Risk] is mitigated by proper data splitting and evaluation metrics.
# [No Hyperparameter Tuning] - as per instructions, using default RandomForestClassifier
clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(random_state=42))])

# Training model
clf_pipeline.fit(X_train, y_train)
y_pred = clf_pipeline.predict(X_test)
y_proba = clf_pipeline.predict_proba(X_test)[:, 1] # Get probabilities for ROC AUC

# Evaluating model - [Limited Model Evaluation]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba)) # Add ROC AUC score for more robust evaluation

# --- End of Modifications ---