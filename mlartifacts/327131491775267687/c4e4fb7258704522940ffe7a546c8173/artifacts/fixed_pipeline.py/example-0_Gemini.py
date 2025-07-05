import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

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
data = data.drop(['Name', 'Cabin', 'Ticket'], axis=1)

# [Covariate_Shift] Removed artificial creation of covariate shift by sampling classes unequally.
# Using the original distribution of the data.

# Handling missing values
# [Missing_Value_Imputation_Fare] Corrected Fare imputation to use the median of 'Fare' itself, not 'Age'.
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Fare'] = data['Fare'].fillna(data['Fare'].median()) # Corrected line

# Encoding categorical variables
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
# [Missing_Values_After_Imputation] Ensured Embarked is handled before encoding.
# The previous code handled Embarked correctly, but the overall issue stemmed from Fare.
data['Embarked'] = le.fit_transform(data['Embarked'])

# Checking for any remaining missing values after imputation and encoding
# [Missing_Values_After_Imputation] This check will now pass as all specified missing values are handled.
print("Missing values after imputation and encoding:")
print(data.isnull().sum())

# Splitting the dataset into train and test sets
X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Checking for any remaining missing values in train and test sets
# [Missing_Values_After_Imputation] These checks will now pass as the original DataFrame was fully imputed.
print("Missing values in X_train:")
print(X_train.isnull().sum())
print("Missing values in X_test:")
print(X_test.isnull().sum())

# Training a simple classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Prediction and evaluation
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report: {classification_report(y_test, y_pred)}")