# Summary of fixes:
# - Changed df_encoded to use drop_first=False to include all dummy variables for accurate correlation analysis. #FIXED
# - Modified identify_proxy_attributes to map dummy-coded proxy attributes back to their original feature names. #FIXED
# - Updated identification of encoded_columns to only include those starting with protected_attribute + '_' prefix. #FIXED
# - Configured OneHotEncoder with sparse=False for compatibility with RandomForestClassifier. #FIXED

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

def identify_proxy_attributes(df, protected_attributes, correlation_threshold=0.8):
    proxy_attributes = set()
    
    # generate all dummy variables
    df_encoded = pd.get_dummies(df, drop_first=False)  #FIXED
    
    corr_matrix = df_encoded.corr().abs()
    
    for protected_attribute in protected_attributes:
        if protected_attribute in df.columns:
            # only dummy columns for this protected attribute
            encoded_columns = [col for col in df_encoded.columns if col.startswith(f"{protected_attribute}_")]  #FIXED
            for encoded_col in encoded_columns:
                correlated = corr_matrix[encoded_col][corr_matrix[encoded_col] > correlation_threshold].index.tolist()  #FIXED
                for col in correlated:  #FIXED
                    if col in df.columns:  #FIXED
                        proxy_attributes.add(col)  #FIXED
                    elif "_" in col:  #FIXED
                        base = col.split("_")[0]  #FIXED
                        if base in df.columns and base not in protected_attributes:  #FIXED
                            proxy_attributes.add(base)  #FIXED
    
    return list(proxy_attributes)

protected_attributes = ['race', 'gender']

X_train, X_test, y_train, y_test = train_test_split(data.drop('salary', axis=1), data['salary'], test_size=0.2, random_state=42)

proxy_attributes = identify_proxy_attributes(X_train, protected_attributes)

# drop protected and proxy features
features_to_include = [col for col in X_train.columns if col not in protected_attributes + proxy_attributes]
X_train = X_train[features_to_include]
X_test = X_test[features_to_include]

categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_columns)  #FIXED
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

print(f"Removed proxy attributes: {proxy_attributes}")