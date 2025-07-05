# Summary of fixes:
# - Restricted replacement of 'Medium' to the 'score_text' column only.
# - Avoided data leakage by splitting data before fitting transformers.
# - Replaced label_binarize with LabelEncoder for proper 1D label encoding.
# - Combined preprocessing and classification into a single Pipeline.
# - Added stratify parameter to train_test_split to maintain class distribution.

import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assume fixed.py is executed from the project root
raw_data_file = os.path.join(os.getcwd(), "datasets", "compas_scores", "compas-scores-two-years.csv")
raw_data = pd.read_csv(raw_data_file)

# Select only relevant columns
raw_data = raw_data[
    ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count',
     'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid',
     'c_jail_in', 'c_jail_out']
]

# Filter data
raw_data = raw_data[
    (raw_data['days_b_screening_arrest'] <= 30) &
    (raw_data['days_b_screening_arrest'] >= -30) &
    (raw_data['is_recid'] != -1) &
    (raw_data['c_charge_degree'] != 'O') &
    (raw_data['score_text'] != 'N/A')
]

# Restrict replacement to 'score_text' column
raw_data['score_text'] = raw_data['score_text'].replace('Medium', 'Low')

# Define features and label
X = raw_data[['is_recid', 'age']]
y = raw_data['score_text']

# Split before preprocessing to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Encode labels to 0/1
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# Preprocessing pipelines
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))
])

preprocessor = ColumnTransformer([
    ('cat', cat_pipeline, ['is_recid']),
    ('num', num_pipeline, ['age'])
])

# Combine preprocessing and classification
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train and evaluate
pipeline.fit(X_train, y_train_enc)
print("Training accuracy:", pipeline.score(X_train, y_train_enc))
print("Test accuracy:", pipeline.score(X_test, y_test_enc))

y_pred = pipeline.predict(X_test)
print(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_, zero_division=0))