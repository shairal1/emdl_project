# Summary of fixes:
# - Changed target variable from 'Income' to 'Diabetes_binary'
# - Added stratify parameter to train_test_split for class distribution preservation
# - Scaled features using StandardScaler before resampling and modeling
# - Removed unused import 'resample'
# - Wrapped script execution in main guard

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#FIXED
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

#FIXED
def main():
    #FIXED
    project_root = get_project_root()

    raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "binary_health_indicators.csv")
    data = pd.read_csv(raw_data_file)

    #FIXED
    print("Class distribution in raw data:\n", data['Diabetes_binary'].value_counts(normalize=True).round(2))

    #FIXED
    X = data.drop('Diabetes_binary', axis=1)
    #FIXED
    y = data['Diabetes_binary']
    #FIXED
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    #FIXED
    scaler = StandardScaler()
    #FIXED
    X_train_scaled = scaler.fit_transform(X_train)
    #FIXED
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    #FIXED
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print("Class distribution after resampling:\n",
          pd.Series(y_train_resampled).value_counts(normalize=True).round(2))

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_resampled, y_train_resampled)

    #FIXED
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

#FIXED
if __name__ == "__main__":
    #FIXED
    main()
