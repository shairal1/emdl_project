# fixed.py

# Summary of fixes:
# - Removed sys.path manipulation and dependency on utils.get_project_root; use script directory for data path.
# - Imputed missing values: 'Age' with mean, 'Embarked' with mode.
# - Performed one-hot encoding after imputation.
# - Used stratified train/test split to maintain class distribution.
# - Enabled parallel processing in RandomForestClassifier with n_jobs=-1.
# - Included target names in classification report for clarity.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Define path to the dataset relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_file = os.path.join(script_dir, "datasets", "titanic", "data.csv")

# Load data
data = pd.read_csv(raw_data_file)

# Drop irrelevant features
X = data.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
y = data['Survived']

# Impute missing values
X['Age'] = X['Age'].fillna(X['Age'].mean())
if X['Embarked'].isnull().any():
    X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Initialize and train the classifier
clf = RandomForestClassifier(random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=[str(c) for c in label_encoder.classes_]))
