# Summary of fixes:
# - Imputed missing values for 'Age', 'Fare', and 'Embarked' before encoding.
# - Added imputation for 'Fare' using median.
# - Added imputation for 'Embarked' using most frequent category (mode).
# - Ensured categorical encoding via get_dummies occurs after imputation.

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "titanic", "data.csv")
data = pd.read_csv(raw_data_file)

X = data.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
y = data['Survived']

#FIXED
X['Age'] = X['Age'].fillna(X['Age'].mean())  
#FIXED
X['Fare'] = X['Fare'].fillna(X['Fare'].median())  
#FIXED
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])  
#FIXED
X = pd.get_dummies(X, columns=['Sex', 'Embarked'])  

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
