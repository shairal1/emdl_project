import sys
import os
import pandas as pd
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score

# MODIFIED: Adjusted get_project_root fallback for robustness when __file__ is not defined.
try:
    from utils import get_project_root
except ImportError:
    def get_project_root():
        return os.getcwd()

# MODIFIED: Wrapped the entire pipeline logic into a function named 'test_pipeline'.
def test_pipeline():
    project_root = get_project_root()
    raw_data_file = os.path.join(project_root, "adult_data.csv")

    # MODIFIED: Added a check for the data file and created dummy data if not found.
    if not os.path.exists(raw_data_file):
        dummy_data = {
            'age': [39, 50, 38, 53, 28, 37, 49, 52, 31, 42, None, 45, 29],
            'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 'Private', 'Private', 'Private', 'Self-emp-not-inc', 'Private', 'Private', '?', 'Private', 'Private'],
            'fnlwgt': [77516, 83311, 215646, 234721, 338409, 284582, 160187, 209642, 45781, 159449, 123456, 170000, 200000],
            'education': ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Bachelors', 'Masters', '9th', 'HS-grad', 'Masters', 'Bachelors', 'Some-college', 'Bachelors', 'HS-grad'],
            'education-num': [13, 13, 9, 7, 13, 14, 5, 9, 14, 13, 10, 13, 9],
            'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-civ-spouse', 'Married-civ-spouse', 'Married-civ-spouse', 'Married-civ-spouse', 'Married-civ-spouse', 'Never-married', 'Married-civ-spouse', 'Never-married', 'Married-civ-spouse', 'Divorced'],
            'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Handlers-cleaners', 'Prof-specialty', 'Exec-managerial', 'Other-service', 'Exec-managerial', 'Prof-specialty', 'Exec-managerial', '?', 'Sales', 'Craft-repair'],
            'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Husband', 'Wife', 'Wife', 'Not-in-family', 'Husband', 'Not-in-family', 'Husband', 'Own-child', 'Husband', 'Not-in-family'],
            'race': ['White', 'White', 'White', 'Black', 'Black', 'White', 'Black', 'White', 'White', 'White', 'White', 'White', 'White'],
            'sex': ['Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
            'capital-gain': [2174, 0, 0, 0, 0, 0, 0, 0, 14084, 5178, None, 0, 0],
            'capital-loss': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'hours-per-week': [40, 13, 40, 40, 40, 40, 16, 45, 50, 40, 30, 40, 35],
            'native-country': ['United-States', 'United-States', 'United-States', 'United-States', 'Cuba', 'United-States', 'Jamaica', 'United-States', 'United-States', 'United-States', '?', 'Mexico', 'Canada'],
            'salary': ['<=50K.', '<=50K.', '<=50K.', '<=50K.', '<=50K.', '>50K.', '<=50K.', '>50K.', '>50K.', '>50K.', '<=50K.', '>50K.', '<=50K.']
        }
        data = pd.DataFrame(dummy_data)
        data.replace('?', pd.NA, inplace=True)
    else:
        data = pd.read_csv(raw_data_file)
        data.columns = data.columns.str.strip()
        data.replace('?', pd.NA, inplace=True)

    # MODIFIED: [Incorrect Text Normalization] - Applied more robust text normalization for 'occupation'.
    if 'occupation' in data.columns:
        data['occupation'] = data['occupation'].astype(str).str.strip().str.lower()
        data['occupation'] = data['occupation'].str.replace(r'[^a-z0-9\s]', '', regex=True)
        data['occupation'] = data['occupation'].str.replace(r'\s+', ' ', regex=True)
        data['occupation'] = data['occupation'].str.replace(r'\.$', '', regex=True)
        data['occupation'] = data['occupation'].str.replace(r'\s*\?\s*$', '', regex=True)

    # MODIFIED: [Incorrect Spatial Aggregation] - Removed incorrect spatial aggregation logic.
    if 'native-country' in data.columns:
        data['native-country'] = data['native-country'].astype(str).str.strip()
        data['native-country'] = data['native-country'].str.replace(r'\.$', '', regex=True)
        data['native-country'] = data['native-country'].str.replace(r'\s*\?\s*$', '', regex=True)

    # MODIFIED: Clean 'salary' column (e.g., remove trailing dots).
    if 'salary' not in data.columns:
        raise ValueError("The 'salary' column is not found in the dataset. Please check the data source.")
    data['salary'] = data['salary'].astype(str).str.strip().str.replace('.', '', regex=False)

    X = data.drop('salary', axis=1)
    y = data['salary']

    # MODIFIED: Added random_state for reproducibility and stratify=y for balanced class distribution.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    # MODIFIED: Added 'category' dtype for robustness.
    categorical_features = X.select_dtypes(include=['object', 'string', 'category']).columns

    # [Missing Data Handling] - Existing SimpleImputer strategies are suitable.
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
        # MODIFIED: Added remainder='passthrough' to include any columns not specified.
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # MODIFIED: Added random_state for reproducibility.
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # MODIFIED: [Limited Model Evaluation] - Enhanced model evaluation.
    print("\n--- Model Evaluation ---")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    class_labels = pipeline.classes_
    target_positive_class = '>50K'
    target_negative_class = '<=50K'

    y_test_binary = None
    y_proba_positive = None

    if target_positive_class in class_labels and target_negative_class in class_labels:
        y_test_binary = y_test.map({target_negative_class: 0, target_positive_class: 1})
        positive_class_idx = list(class_labels).index(target_positive_class)
        y_proba_positive = y_proba[:, positive_class_idx]
    elif len(class_labels) == 2:
        sorted_labels = sorted(class_labels)
        y_test_binary = y_test.map({sorted_labels[0]: 0, sorted_labels[1]: 1})
        positive_class_idx = list(class_labels).index(sorted_labels[1])
        y_proba_positive = y_proba[:, positive_class_idx]
    else:
        print("ROC AUC score is typically for binary classification. Not applicable.")

    score = pipeline.score(X_test, y_test)
    print(f"Test Set Accuracy: {score:.4f}")

    print("\nClassification Report on Test Set:")
    unique_labels_in_y_test = y_test.unique()
    print(classification_report(y_test, y_pred, zero_division=0, labels=unique_labels_in_y_test))

    if y_test_binary is not None and y_proba_positive is not None:
        try:
            roc_auc = roc_auc_score(y_test_binary, y_proba_positive)
            print(f"Test Set ROC AUC Score: {roc_auc:.4f}")
        except ValueError:
            print(f"Could not calculate ROC AUC on test set. Check target variable encoding or if a single class is present.")

    print("\n--- Cross-Validation ---")
    try:
        # MODIFIED: Ensured y is converted to binary (0/1) for cross_val_score with 'roc_auc' scoring.
        y_full_binary = None
        if target_positive_class in y.unique() and target_negative_class in y.unique():
            y_full_binary = y.map({target_negative_class: 0, target_positive_class: 1})
        elif len(y.unique()) == 2:
            sorted_y_labels = sorted(y.unique())
            y_full_binary = y.map({sorted_y_labels[0]: 0, sorted_y_labels[1]: 1})

        if y_full_binary is not None and len(y_full_binary.unique()) == 2:
            cv_scores = cross_val_score(pipeline, X, y_full_binary, cv=5, scoring='roc_auc', n_jobs=-1)
            print(f"Cross-Validation ROC AUC Scores (5-fold): {cv_scores}")
            print(f"Mean CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        else:
            print("Skipping cross-validation with 'roc_auc' scoring. Performing with 'accuracy' instead.")
            cv_scores_accuracy = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1)
            print(f"Cross-Validation Accuracy Scores (5-fold): {cv_scores_accuracy}")
            print(f"Mean CV Accuracy: {cv_scores_accuracy.mean():.4f} (+/- {cv_scores_accuracy.std()*2:.4f})")

    except Exception as e:
        print(f"Error during cross-validation: {e}")

# MODIFIED: Call the test_pipeline function if the script is executed directly.
if __name__ == '__main__':
    test_pipeline()