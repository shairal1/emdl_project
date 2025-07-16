# Summary of fixes:
# - Removed unused imports: sys, cross_val_score, and Pipeline from sklearn.pipeline
# - Simplified path handling using pathlib.Path; removed sys.path manipulation and utils import
# - Added shuffle=True and random_state=42 to StratifiedKFold for reproducibility
# - Changed scoring metric from 'accuracy' to 'roc_auc' for more appropriate evaluation on imbalanced data

from pathlib import Path  #FIXED
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV  #FIXED
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

#FIXED script_dir = Path(__file__).resolve().parent
#FIXED project_root = script_dir.parent
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
raw_data_file = project_root / "datasets" / "diabetes_indicator" / "5050_split.csv"

#FIXED try/except block to provide clear error if file is missing
try:
    data = pd.read_csv(raw_data_file)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found: {raw_data_file}")

pipeline = ImbPipeline([
    ('sampling', RandomUnderSampler(random_state=42)),
    ('feature_selection', SelectKBest(f_classif, k=10)),
    ('classifier', RandomForestClassifier(random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  #FIXED

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20]
}

#FIXED scoring changed from 'accuracy' to 'roc_auc'
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(data.drop('Diabetes_binary', axis=1), data['Diabetes_binary'])

print("Best parameters:", grid_search.best_params_)
#FIXED updated output to reflect new metric
print("Best cross-validation ROC AUC score:", grid_search.best_score_)