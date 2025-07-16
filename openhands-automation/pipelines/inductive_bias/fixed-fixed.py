# Summary of fixes:
# - Properly handle missing values represented by '?' in adult dataset using na_values in read_csv
# - Strip whitespace from categorical columns to avoid inconsistent labels
# - Force OneHotEncoder to return dense arrays (sparse=False) to avoid sparse/dense mixing issues
# - Add random_state=42 to LogisticRegression for reproducibility
# - Set n_jobs=-1 in GridSearchCV to utilize all CPU cores for faster search
# - Encapsulate script execution in a main() function and add __main__ guard

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#FIXED
def main():
    # Determine project root
    #FIXED
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #FIXED
    parent_dir = os.path.dirname(current_dir)
    #FIXED
    sys.path.append(parent_dir)
    #FIXED
    from utils import get_project_root
    #FIXED
    project_root = get_project_root()

    #FIXED
    raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
    #FIXED
    data = pd.read_csv(raw_data_file, na_values=['?',' ?'])

    #FIXED
    object_cols = data.select_dtypes(include=['object']).columns.tolist()
    #FIXED
    data[object_cols] = data[object_cols].apply(lambda col: col.str.strip())

    X = data.drop('salary', axis=1)
    y = data['salary']

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                #FIXED
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ]), categorical_features),

            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), X.select_dtypes(include=['number']).columns.tolist())
        ])

    #FIXED
    clf = LogisticRegression(max_iter=1000, random_state=42)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    param_grid = {
        'classifier__C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    }
    #FIXED
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print(f"Best regularization parameter (C): {grid_search.best_params_['classifier__C']}")

#FIXED
if __name__ == '__main__':
    main()
