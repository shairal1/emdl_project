# fixed.py
#
# Summary of fixes:
# - Drop 'PassengerId' column to avoid using it as a feature.
# - Corrected 'Fare' imputation: fill missing 'Fare' values with median of 'Fare', not incorrectly using 'Age'.
# - Use separate LabelEncoder instances for 'Sex' and 'Embarked' to avoid encoding conflicts.
# - Shuffle the dataset after concatenation to randomize order.
# - Perform train/test split before imputation and encoding to prevent data leakage.
# - Impute missing values on training set and apply to test set to avoid peeking at test data.

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Add project root to path and import utility
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import get_project_root

# Load data
project_root = get_project_root()
raw_data_file = os.path.join(project_root, "datasets", "titanic", "data.csv")
data = pd.read_csv(raw_data_file)

# Drop unused or leaking columns
cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data = data.drop(cols_to_drop, axis=1)

# Balance classes by downsampling the majority class
df_class_0 = data[data['Survived'] == 0].sample(frac=0.6, random_state=42)
df_class_1 = data[data['Survived'] == 1]
df_balanced = pd.concat([df_class_0, df_class_1]).sample(frac=1, random_state=42)

# Separate features and target
X = df_balanced.drop('Survived', axis=1)
y = df_balanced['Survived']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Make explicit copies to avoid SettingWithCopyWarning
X_train = X_train.copy()
X_test = X_test.copy()

# Impute missing values using training set statistics
age_median = X_train['Age'].median()
fare_median = X_train['Fare'].median()
embarked_mode = X_train['Embarked'].mode()[0]

X_train['Age'].fillna(age_median, inplace=True)
X_test['Age'].fillna(age_median, inplace=True)

X_train['Fare'].fillna(fare_median, inplace=True)
X_test['Fare'].fillna(fare_median, inplace=True)

X_train['Embarked'].fillna(embarked_mode, inplace=True)
X_test['Embarked'].fillna(embarked_mode, inplace=True)

# Encode categorical variables with separate LabelEncoders
le_sex = LabelEncoder()
X_train['Sex'] = le_sex.fit_transform(X_train['Sex'])
X_test['Sex'] = le_sex.transform(X_test['Sex'])

le_embarked = LabelEncoder()
X_train['Embarked'] = le_embarked.fit_transform(X_train['Embarked'])
X_test['Embarked'] = le_embarked.transform(X_test['Embarked'])

# Verify no missing values remain
print("Missing values in training set:")
print(X_train.isnull().sum())
print("Missing values in test set:")
print(X_test.isnull().sum())

# Train a RandomForest model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
