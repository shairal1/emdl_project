# Summary of fixes:
# - Fixed label encoding: replaced label_binarize with binary mapping.
# - Dropped 'dob' column from features.
# - Added preprocessing pipelines for categorical ('sex', 'c_charge_degree', 'race', 'is_recid') and numeric features ('priors_count','days_b_screening_arrest','decile_score','two_year_recid','c_jail_in','c_jail_out') with imputation and encoding/scaling.
# - Updated ColumnTransformer to include all specified features and set remainder='drop'.
# - Added solver='liblinear' to LogisticRegression for compatibility.

import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer  #FIXED
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
raw_data = pd.read_csv(raw_data_file)

train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)

columns_to_use = ['sex', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
                  'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']  #FIXED

train_data = train_data[columns_to_use]
test_data = test_data[columns_to_use]

print("Shape of training data:", train_data.shape)
print("Shape of testing data:", test_data.shape)

train_data = train_data.replace('Medium', "Low")
test_data = test_data.replace('Medium', "Low")

# Target encoding: map 'High' to 1, 'Low' to 0 instead of label_binarize  #FIXED
train_labels = (train_data['score_text'] == 'High').astype(int)  #FIXED
test_labels = (test_data['score_text'] == 'High').astype(int)  #FIXED

train_data = train_data.drop(columns=['score_text'])
test_data = test_data.drop(columns=['score_text'])

train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# Preprocessing pipelines  #FIXED
cat_impute_and_onehot = Pipeline([  #FIXED
    ('imputer', SimpleImputer(strategy='most_frequent')),  #FIXED
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  #FIXED
])  #FIXED

age_impute_and_bin = Pipeline([  #FIXED
    ('imputer', SimpleImputer(strategy='mean')),  #FIXED
    ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))  #FIXED
])  #FIXED

num_impute_and_scale = Pipeline([  #FIXED
    ('imputer', SimpleImputer(strategy='mean')),  #FIXED
    ('scaler', StandardScaler())  #FIXED
])  #FIXED

featurizer = ColumnTransformer(transformers=[  #FIXED
    ('cat_basic', cat_impute_and_onehot, ['sex', 'c_charge_degree', 'race']),  #FIXED
    ('cat_recid', cat_impute_and_onehot, ['is_recid']),  #FIXED
    ('age_bins', age_impute_and_bin, ['age']),  #FIXED
    ('num', num_impute_and_scale, ['priors_count', 'days_b_screening_arrest', 'decile_score',  #FIXED
                                   'two_year_recid', 'c_jail_in', 'c_jail_out'])  #FIXED
], remainder='drop')  #FIXED

pipeline = Pipeline(steps=[('featurizer', featurizer),
                           ('classifier', LogisticRegression(solver='liblinear'))])  #FIXED

pipeline.fit(train_data, train_labels)

print("Model score:", pipeline.score(test_data, test_labels))

print(classification_report(test_labels, pipeline.predict(test_data), zero_division=0))