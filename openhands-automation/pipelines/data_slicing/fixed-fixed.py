# Summary of fixes:
# - Changed target labels from score_text to two_year_recid to predict actual recidivism.
# - Removed mapping of 'Medium' to 'Low' and the label_binarize import.
# - Extracted two_year_recid as numeric labels.
# - Dropped both score_text and two_year_recid from feature set.
# - Removed unused StandardScaler import.
# - Marked modified lines with #FIXED.

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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
raw_data = pd.read_csv(raw_data_file)

train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)

columns_to_use = ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
                  'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']

train_data = train_data[columns_to_use]
test_data = test_data[columns_to_use]

print("Shape of training data:", train_data.shape)
print("Shape of testing data:", test_data.shape)

# Removed mapping of 'Medium' to 'Low' since score_text is no longer used as label

#FIXED
train_labels = train_data['two_year_recid'].astype(int).values
#FIXED
test_labels = test_data['two_year_recid'].astype(int).values

#FIXED
train_data = train_data.drop(columns=['score_text', 'two_year_recid'])
#FIXED
test_data = test_data.drop(columns=['score_text', 'two_year_recid'])

train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

impute1_and_onehot = Pipeline([('imputer1', SimpleImputer(strategy='most_frequent')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])
impute2_and_bin = Pipeline([('imputer2', SimpleImputer(strategy='mean')),
                            ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))])

featurizer = ColumnTransformer(transformers=[
    ('impute1_and_onehot', impute1_and_onehot, ['is_recid']),
    ('impute2_and_bin', impute2_and_bin, ['age'])
])

pipeline = Pipeline(steps=[('featurizer', featurizer),
    ('classifier', LogisticRegression())
])

pipeline.fit(train_data, train_labels)

print("Model score:", pipeline.score(test_data, test_labels))

print(classification_report(test_labels, pipeline.predict(test_data), zero_division=0))

slices = test_data.groupby(['race', 'sex'])

for slice_name, slice_df in slices:
    slice_indices = slice_df.index
    slice_y_true = test_labels[slice_indices]
    slice_y_pred = pipeline.predict(slice_df)
    print(f"Performance for slice {slice_name}:")
    print(classification_report(slice_y_true, slice_y_pred, zero_division=0))