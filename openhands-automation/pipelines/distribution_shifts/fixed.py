"""
Summary of fixes:
- Dropped 'PassengerId' column to remove non-predictive ID feature.
- Fixed incorrect fillna on 'Fare' column.
- Avoided data leakage by splitting dataset before imputation and encoding.
- Computed imputation values (median/mode) on training set only.
- Used separate LabelEncoders for 'Sex' and 'Embarked' and applied transform on test set.
- Shuffled the combined dataset after class sampling.
"""

import os
import sys
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
#FIXED Dropped PassengerId column
data = data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)

df_class_0 = data[data['Survived'] == 0].sample(frac=0.6, random_state=42)
df_class_1 = data[data['Survived'] == 1]

df_shifted = pd.concat([df_class_0, df_class_1])
#FIXED Shuffled dataset after sampling to randomize order
df_shifted = df_shifted.sample(frac=1, random_state=42)

X = df_shifted.drop('Survived', axis=1)
y = df_shifted['Survived']

#FIXED Split data before imputation to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#FIXED Compute training set statistics for imputation
age_median = X_train['Age'].median()
fare_median = X_train['Fare'].median()
embarked_mode = X_train['Embarked'].mode()[0]

#FIXED Impute numerical features
X_train['Age'] = X_train['Age'].fillna(age_median)
X_test['Age'] = X_test['Age'].fillna(age_median)
X_train['Fare'] = X_train['Fare'].fillna(fare_median)
X_test['Fare'] = X_test['Fare'].fillna(fare_median)

#FIXED Impute categorical features
X_train['Embarked'] = X_train['Embarked'].fillna(embarked_mode)
X_test['Embarked'] = X_test['Embarked'].fillna(embarked_mode)

#FIXED Encode categorical features with separate encoders
le_sex = LabelEncoder()
X_train['Sex'] = le_sex.fit_transform(X_train['Sex'])
X_test['Sex'] = le_sex.transform(X_test['Sex'])

le_embarked = LabelEncoder()
X_train['Embarked'] = le_embarked.fit_transform(X_train['Embarked'])
X_test['Embarked'] = le_embarked.transform(X_test['Embarked'])

print("Missing values after preprocessing:")
print(X_train.isnull().sum())
print(X_test.isnull().sum())

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report: {classification_report(y_test, y_pred)}")
