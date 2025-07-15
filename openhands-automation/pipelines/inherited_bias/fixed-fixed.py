#FIXED Summary of fixes:
#FIXED - Handled missing values: replaced '?' with NaN and dropped missing rows.
#FIXED - Removed 'race' from feature set to avoid using the sensitive attribute as a predictor.
#FIXED - Changed OneHotEncoder parameter 'sparse_output' to 'sparse=False' for compatibility.
#FIXED - Extracted sensitive_features from X_train before dropping 'race'.
#FIXED - Ensured training and testing feature sets exclude the 'race' column.
#FIXED - Added random_state to LogisticRegression for reproducibility.
#FIXED - Included target_names in classification_report for clarity.

import os
import sys
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

warnings.simplefilter(action='ignore', category=FutureWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
#FIXED
data = pd.read_csv(raw_data_file, skipinitialspace=True, na_values=['?'])
#FIXED
data.dropna(inplace=True)

X = data.drop('salary', axis=1)
y = data['salary']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

race_encoder = LabelEncoder()
X['race'] = race_encoder.fit_transform(X['race'])

#FIXED
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#FIXED
sensitive_train = X_train['race']

#FIXED
encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
#FIXED
cat_cols = X_train.select_dtypes(include=['object']).columns

#FIXED
X_train_encoded = encoder.fit_transform(X_train[cat_cols])
#FIXED
X_test_encoded = encoder.transform(X_test[cat_cols])

X_train_encoded_df = pd.DataFrame(
    X_train_encoded,
    columns=encoder.get_feature_names_out(cat_cols)
)
X_test_encoded_df = pd.DataFrame(
    X_test_encoded,
    columns=encoder.get_feature_names_out(cat_cols)
)

#FIXED
X_train_numeric = X_train.select_dtypes(exclude=['object']).drop(columns=['race']).reset_index(drop=True)
#FIXED
X_test_numeric = X_test.select_dtypes(exclude=['object']).drop(columns=['race']).reset_index(drop=True)

X_train_final = pd.concat([X_train_numeric, X_train_encoded_df.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_test_numeric, X_test_encoded_df.reset_index(drop=True)], axis=1)

X_train_final.columns = X_train_final.columns.astype(str)
X_test_final.columns = X_test_final.columns.astype(str)

#FIXED
model = LogisticRegression(max_iter=10000, random_state=42)
mitigator = ExponentiatedGradient(
    estimator=model,
    constraints=DemographicParity()
)

#FIXED
mitigator.fit(X_train_final, y_train, sensitive_features=sensitive_train)

#FIXED
y_pred = mitigator.predict(X_test_final)
#FIXED
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))