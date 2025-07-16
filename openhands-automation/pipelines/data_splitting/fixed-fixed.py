# Summary of fixes:
# - Applied identical preprocessing filters to test_data to mirror train_data filtering.
# - Replaced global DataFrame.replace with column-specific replacement on 'score_text' to avoid unintended replacements.
# - Flattened label arrays using .ravel() for compatibility with scikit-learn estimators and metrics.
# - Used ColumnTransformer.transform for test_data instead of fit_transform to prevent data leakage.

import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, label_binarize
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

train_data = train_data[
    ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text',
     'priors_count', 'days_b_screening_arrest', 'decile_score',
     'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']
]
test_data = test_data[
    ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text',
     'priors_count', 'days_b_screening_arrest', 'decile_score',
     'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']
]

train_data = train_data[(train_data['days_b_screening_arrest'] <= 30) & (train_data['days_b_screening_arrest'] >= -30)]
train_data = train_data[train_data['is_recid'] != -1]
train_data = train_data[train_data['c_charge_degree'] != "O"]
train_data = train_data[train_data['score_text'] != 'N/A']
#FIXED
test_data = test_data[(test_data['days_b_screening_arrest'] <= 30) & (test_data['days_b_screening_arrest'] >= -30)]
#FIXED
test_data = test_data[test_data['is_recid'] != -1]
#FIXED
test_data = test_data[test_data['c_charge_degree'] != "O"]
#FIXED
test_data = test_data[test_data['score_text'] != 'N/A']

#FIXED
train_data['score_text'] = train_data['score_text'].replace('Medium', 'Low')
#FIXED
test_data['score_text'] = test_data['score_text'].replace('Medium', 'Low')

#FIXED
train_labels = label_binarize(train_data['score_text'], classes=['High', 'Low']).ravel()
#FIXED
test_labels = label_binarize(test_data['score_text'], classes=['High', 'Low']).ravel()

impute1_and_onehot = Pipeline([('imputer1', SimpleImputer(strategy='most_frequent')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])
impute2_and_bin = Pipeline([('imputer2', SimpleImputer(strategy='mean')),
                            ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))])

featurizer = ColumnTransformer(transformers=[
    ('impute1_and_onehot', impute1_and_onehot, ['is_recid']),
    ('impute2_and_bin', impute2_and_bin, ['age'])
])

train_data = featurizer.fit_transform(train_data)
#FIXED
test_data = featurizer.transform(test_data)

print("Shape of training data:", train_data.shape)
print("Shape of testing data:", test_data.shape)

pipeline = Pipeline([('classifier', LogisticRegression())])

#FIXED
pipeline.fit(train_data, train_labels)
#FIXED
print(pipeline.score(test_data, test_labels))

print(classification_report(test_labels, pipeline.predict(test_data), zero_division=0))
