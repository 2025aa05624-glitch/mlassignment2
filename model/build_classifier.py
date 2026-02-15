import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib
import os

# Make directory for saved classifiers
os.makedirs('model', exist_ok=True)

# Load digits dataset
digits = load_digits()
features = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(digits.data.shape[1])])
target = digits.target

# All features numerical
num_features = list(features.columns)

# Preprocessing: scale
prep_steps = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), num_features)
    ])

# Split
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=7)

# Classifiers
classifiers = {
    'LogReg': Pipeline([('prep', prep_steps), ('model', LogisticRegression(max_iter=1000))]),
    'DecTree': Pipeline([('prep', prep_steps), ('model', DecisionTreeClassifier(random_state=7))]),
    'KNeigh': Pipeline([('prep', prep_steps), ('model', KNeighborsClassifier())]),
    'GausNB': Pipeline([('prep', prep_steps), ('model', GaussianNB())]),
    'RandFor': Pipeline([('prep', prep_steps), ('model', RandomForestClassifier(random_state=7))]),
    'XGB': Pipeline([('prep', prep_steps), ('model', XGBClassifier(eval_metric='mlogloss', random_state=7))])
}

# Train, eval, save
perf_dict = {}
for clf_name, clf_pipe in classifiers.items():
    clf_pipe.fit(features_train, target_train)
    pred_labels = clf_pipe.predict(features_test)
    prob_scores = clf_pipe.predict_proba(features_test)
    
    acc_val = accuracy_score(target_test, pred_labels)
    auc_val = roc_auc_score(target_test, prob_scores, multi_class='ovr')
    prec_val = precision_score(target_test, pred_labels, average='macro')
    rec_val = recall_score(target_test, pred_labels, average='macro')
    f1_val = f1_score(target_test, pred_labels, average='macro')
    mcc_val = matthews_corrcoef(target_test, pred_labels)
    
    perf_dict[clf_name] = {'Accuracy': acc_val, 'AUC': auc_val, 'Precision': prec_val, 'Recall': rec_val, 'F1': f1_val, 'MCC': mcc_val}
    
    joblib.dump(clf_pipe, f'model/{clf_name}.pkl')

print("Performance:")
for name, vals in perf_dict.items():
    print(f"{name}: {vals}")

# Save test for app
test_set = features_test.copy()
test_set['target'] = target_test
test_set.to_csv('test_dataset.csv', index=False)