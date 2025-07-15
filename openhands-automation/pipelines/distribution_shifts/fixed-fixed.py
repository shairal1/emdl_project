# Summary of fixes:
# 1. Dropped PassengerId to remove non-predictive identifier.
# 2. Fixed fillna for Fare (was incorrectly using Age).
# 3. Used separate LabelEncoders for 'Sex' and 'Embarked'.
# 4. Imported numpy for weighted resampling logic.
# 5. Replaced unsupported resample weights with sample_weight in RandomForestClassifier.
# 6. Ensured consistent scaling by training on scaled features.

import os
import sys
import pandas as pd
import numpy as np  #FIXED
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.kernel_ridge import KernelRidge

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "titanic", "data.csv")
data = pd.read_csv(raw_data_file)

# Drop identifier columns
data = data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)  #FIXED

df_class_0 = data[data['Survived'] == 0].sample(frac=0.6, random_state=42)
df_class_1 = data[data['Survived'] == 1]

df_shifted = pd.concat([df_class_0, df_class_1])

# Impute missing values
df_shifted['Age'] = df_shifted['Age'].fillna(df_shifted['Age'].median())
df_shifted['Embarked'] = df_shifted['Embarked'].fillna(df_shifted['Embarked'].mode()[0])
df_shifted['Fare'] = df_shifted['Fare'].fillna(df_shifted['Fare'].median())  #FIXED

# Encode categorical variables
le_sex = LabelEncoder()  #FIXED
df_shifted['Sex'] = le_sex.fit_transform(df_shifted['Sex'])  #FIXED
le_emb = LabelEncoder()  #FIXED
df_shifted['Embarked'] = le_emb.fit_transform(df_shifted['Embarked'])  #FIXED

# Prepare features and target
X = df_shifted.drop('Survived', axis=1)
y = df_shifted['Survived']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Detect covariate shift using KernelRidge
clf_shift_detection = KernelRidge(kernel='rbf')
clf_shift_detection.fit(X_train_scaled, y_train)
train_preds = clf_shift_detection.predict(X_train_scaled)
test_preds = clf_shift_detection.predict(X_test_scaled)

train_pred_mean = train_preds.mean()
test_pred_mean = test_preds.mean()

# Train final classifier with optional sample weighting
clf = RandomForestClassifier(random_state=42)
if abs(train_pred_mean - test_pred_mean) > 0.1:  # threshold for detecting shift
    print("Covariate shift detected, applying sample weights")
    # Compute class distribution ratios
    train_dist = y_train.value_counts(normalize=True)  #FIXED
    test_dist = y_test.value_counts(normalize=True)    #FIXED
    # Importance weights per sample: p_test(c) / p_train(c)
    sample_weights = (test_dist / train_dist)[y_train].values  #FIXED
    clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)  #FIXED
else:
    clf.fit(X_train_scaled, y_train)  #FIXED

# Evaluate
y_pred = clf.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
