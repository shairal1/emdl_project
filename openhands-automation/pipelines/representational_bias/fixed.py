# fixed.py
#
# Summary of fixes:
# - Replaced raw_data.replace with targeted replacement for score_text only to avoid unintended replacements
# - Switched from label_binarize to LabelEncoder for target encoding, avoiding shape mismatch and flatten issues
# - Ensured ColumnTransformer output is converted to a dense numpy array if sparse, for compatibility with LogisticRegression's lbfgs solver
# - Added max_iter and random_state to LogisticRegression for better convergence and reproducibility

import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
#FIXED changed import to use LabelEncoder instead of label_binarize
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
raw_data = pd.read_csv(raw_data_file)

raw_data = raw_data[
    ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
     'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

#FIXED replace only score_text values to avoid affecting other columnsaw_data['score_text'] = raw_data['score_text'].replace('Medium', 'Low')

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
#FIXED ensure dense numpy array for compatibility with lbfgs solver
if hasattr(prepared_data, 'toarray'):
#FIXED
    prepared_data = prepared_data.toarray()

train_data, test_data, train_labels, test_labels = train_test_split(
    prepared_data,
    raw_data['score_text'],
    test_size=0.2,
    random_state=42
)
print("Shape of training data:", train_data.shape)
print("Shape of testing data:", test_data.shape)

#FIXED encode target labels using LabelEncoder
le = LabelEncoder()
#FIXED
train_labels = le.fit_transform(train_labels)
#FIXED
test_labels = le.transform(test_labels)

#FIXED added max_iter and random_state for reproducibility and convergence
pipeline = Pipeline([('classifier', LogisticRegression(max_iter=1000, random_state=42))])

pipeline.fit(train_data, train_labels)
print("Accuracy", pipeline.score(test_data, test_labels))

print(classification_report(test_labels, pipeline.predict(test_data), zero_division=0))