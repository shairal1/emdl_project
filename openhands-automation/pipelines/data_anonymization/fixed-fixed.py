"""
Summary of fixes:
- Removed sys.path manipulation and util.get_project_root in favor of os.getcwd() (#FIXED)
- Added na_values=['?'] in pd.read_csv and dropped missing rows (#FIXED)
- Stripped whitespace from object columns (#FIXED)
- Instantiated new LabelEncoder for each categorical feature (#FIXED)
- Separate encoder for target variable for reporting (#FIXED)
- Added stratify=y to train_test_split (#FIXED)
- Specified n_estimators=100 in DP_RandomForestClassifier (#FIXED)
- Included target_names in classification_report (#FIXED)
- Wrapped logic in __main__ guard (#FIXED)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from diffprivlib.models import RandomForestClassifier as DP_RandomForestClassifier

def main():
    project_root = os.getcwd()  #FIXED
    raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")  #FIXED

    data = pd.read_csv(raw_data_file, na_values=['?'])  #FIXED
    data.dropna(inplace=True)  #FIXED

    for col in data.select_dtypes(include=['object']):  #FIXED
        data[col] = data[col].str.strip()  #FIXED

    # Anonymize education
    data['education'] = data['education'].apply(lambda x: 'anon' if x in ['Doctorate', 'Masters'] else x)

    y = data['salary']
    X = data.drop(columns=['salary'])

    for col in X.select_dtypes(include=['object']):  #FIXED
        encoder = LabelEncoder()  #FIXED
        X[col] = encoder.fit_transform(X[col])  #FIXED

    target_encoder = LabelEncoder()  #FIXED
    y = target_encoder.fit_transform(y)  #FIXED

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  #FIXED
    )

    model = DP_RandomForestClassifier(n_estimators=100, random_state=42, epsilon=1.0)  #FIXED
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of the corrected pipeline: {accuracy}')
    print("Classification report:")  #FIXED
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))  #FIXED

if __name__ == "__main__":
    main()  #FIXED
