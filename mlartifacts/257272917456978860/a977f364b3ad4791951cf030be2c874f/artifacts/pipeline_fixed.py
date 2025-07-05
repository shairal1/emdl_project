```python
import sys
import os
import pandas as pd
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold # Modified: Added GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin # Modified: Added BaseEstimator, TransformerMixin for custom transformer

# Modified: Custom Transformer for string operations to prevent data leakage
class CustomStringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, transformation_type):
        self.column_name = column_name
        self.transformation_type = transformation_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if self.column_name == 'occupation' and self.transformation_type == 'clean':
            X_transformed[self.column_name] = X_transformed[self.column_name].str.lower()
            X_transformed[self.column_name] = X_transformed[self.column_name].str.replace('-', ' ')
        elif self.column_name == 'native-country' and self.transformation_type == 'aggregate':
            # This is a fixed rule, not learned from data, applied within the pipeline
            X_transformed[self.column_name] = 'North America'
        return X_transformed

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root,"LLM_automation","test_pipeline","adult_data.csv")
data = pd.read_csv(raw_data_file)

# Modified: Removed original data transformations to prevent data leakage.
# These transformations are now handled within the pipeline via CustomStringTransformer.
# data['occupation'] = data['occupation'].str.lower()
# data['occupation'] = data['occupation'].str.replace('-', ' ')
# data['native-country'] = data['native-country'].apply(lambda x: 'North America')

# Modified: Splitting data with stratify to handle data imbalance
X = data.drop('salary', axis=1)
y = data['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('normalizer', Normalizer())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Modified: Added remainder='passthrough' for robustness
)

# Modified: Combined preprocessing with custom string transformers and classifier into a single pipeline
pipeline = Pipeline(steps=[
    # Data Leakage Fix: Apply string transformations within the pipeline
    ('occupation_clean', CustomStringTransformer(column_name='occupation', transformation_type='clean')),
    ('country_aggregate', CustomStringTransformer(column_name='native-country', transformation_type='aggregate')),
    ('preprocessor', preprocessor),
    # Data Imbalance Fix: Added class_weight='balanced'
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Modified: Using GridSearchCV with StratifiedKFold for Cross-Validation and robust evaluation
# Parameter grid for classifier
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__max_depth': [10, 20, None]
}

# Cross-validation strategy for imbalanced data
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy,
                           scoring='f1_weighted', # Modified: Using f1_weighted for imbalanced data
                           n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation F1 score (weighted): {grid_search.best_score_:.2f}")

# Modified: Evaluate best model from GridSearchCV
best_model = grid_search.best_estimator_
score = best_model.score(X_test, y_test)
print(f"Model accuracy on test set: {score:.2f}")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
```