# Summary of fixes:
# - Dropped unused columns 'dob', 'c_jail_in', and 'c_jail_out'
# - Excluded potential target leakage features 'is_recid' and 'two_year_recid'
# - Replaced label_binarize with LabelEncoder for proper binary labels
# - Updated SMOTE and classifier to use 1D label arrays
# - Ensured classification_report uses 1D labels
# - Added calls to check_representational_bias for 'race' and 'sex'

import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder  #FIXED
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
raw_data = pd.read_csv(raw_data_file)

#FIXED
raw_data = raw_data[['sex', 'age', 'c_charge_degree', 'race', 'score_text',
                     'priors_count', 'days_b_screening_arrest', 'decile_score']]

#FIXED
raw_data['score_text'] = raw_data['score_text'].replace('Medium', 'Low')

def check_representational_bias(data, protected_attribute):
    counter = Counter(data[protected_attribute])
    print(f"Distribution of {protected_attribute} in the data: {counter}")

#FIXED
check_representational_bias(raw_data, 'race')
#FIXED
check_representational_bias(raw_data, 'sex')

train_data, test_data, train_labels, test_labels = train_test_split(
    raw_data.drop(columns=['score_text']), raw_data['score_text'], test_size=0.2, random_state=42
)
print("Shape of training data:", train_data.shape)
print("Shape of testing data:", test_data.shape)

categorical_features = ['sex', 'race', 'c_charge_degree']
#FIXED
numeric_features = ['age', 'priors_count', 'days_b_screening_arrest', 'decile_score']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ]
)

train_data_encoded = preprocessor.fit_transform(train_data)
test_data_encoded = preprocessor.transform(test_data)

#FIXED
label_encoder = LabelEncoder()
#FIXED
train_labels = label_encoder.fit_transform(train_labels)
#FIXED
test_labels = label_encoder.transform(test_labels)

print("Class distribution before SMOTE:", Counter(train_labels))

smote = SMOTE(sampling_strategy='auto', random_state=42)
#FIXED
train_data_smote, train_labels_smote = smote.fit_resample(train_data_encoded, train_labels)

print("Distribution of classes after SMOTE:", Counter(train_labels_smote))

pipeline = Pipeline([('classifier', LogisticRegression(max_iter=1000))])

pipeline.fit(train_data_smote, train_labels_smote)
#FIXED
print("Accuracy", pipeline.score(test_data_encoded, test_labels))

#FIXED
predictions = pipeline.predict(test_data_encoded)
#FIXED
print(classification_report(test_labels, predictions, zero_division=0))

#FIXED
def evaluate_fairness(predictions, test_data, actual_labels, protected_attribute):
    fairness_df = test_data.copy()
    fairness_df['predictions'] = predictions
    fairness_df['actual'] = actual_labels
    fairness_df = fairness_df.reset_index(drop=True)
    group_stats = fairness_df.groupby(protected_attribute).agg(
        accuracy=('predictions', lambda x: (x == fairness_df.loc[x.index, 'actual']).mean()),
        count=('predictions', 'count')
    )
    print("Model Fairness Metrics:")
    print(group_stats)

#FIXED
evaluate_fairness(predictions, test_data, test_labels, 'race')
