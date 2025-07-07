# Summary of fixes:
# - Added fallback for __file__ when determining current directory to support interactive environments.
# - Restricted replacement of 'Medium' to only the 'score_text' column to avoid unintended replacements.
# - Removed label_binarize import and its usage; replaced with simple boolean mapping for binary labels.
# - Updated pipeline.fit and pipeline.score calls to use 1D integer label arrays.
# - Enhanced classification_report to include human-readable class names.

import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
#FIXED
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#FIXED
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
#FIXED
except NameError:
#FIXED
    current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
raw_data = pd.read_csv(raw_data_file)

raw_data = raw_data[
    ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
     'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

raw_data = raw_data[(raw_data['days_b_screening_arrest'] <= 30) & (raw_data['days_b_screening_arrest'] >= -30)]
raw_data = raw_data[raw_data['is_recid'] != -1]
raw_data = raw_data[raw_data['c_charge_degree'] != "O"]
raw_data = raw_data[raw_data['score_text'] != 'N/A']

#FIXED
raw_data['score_text'] = raw_data['score_text'].replace('Medium', "Low")

impute1_and_onehot = Pipeline([('imputer1', SimpleImputer(strategy='most_frequent')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])
impute2_and_bin = Pipeline([('imputer2', SimpleImputer(strategy='mean')),
                            ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))])

featurizer = ColumnTransformer(transformers=[
    ('impute1_and_onehot', impute1_and_onehot, ['is_recid']),
    ('impute2_and_bin', impute2_and_bin, ['age'])
])

data_without_labels = raw_data.copy(deep=True)
data_without_labels = data_without_labels.drop(['score_text'], axis=1)

prepared_data = featurizer.fit_transform(data_without_labels)

train_data, test_data, train_labels, test_labels = train_test_split(prepared_data, raw_data['score_text'], test_size=0.2, random_state=42)
print("Shape of training data:", train_data.shape)
print("Shape of testing data:", test_data.shape)

#FIXED
train_labels = (train_labels == 'High').astype(int)
#FIXED
test_labels = (test_labels == 'High').astype(int)

pipeline = Pipeline([('classifier', LogisticRegression())])

#FIXED
pipeline.fit(train_data, train_labels)
#FIXED
print("Accuracy", pipeline.score(test_data, test_labels))

#FIXED
print(classification_report(test_labels, pipeline.predict(test_data), target_names=['Low', 'High'], zero_division=0))
