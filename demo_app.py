import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load classifiers
clf_paths = {
    'LogReg': 'model/LogReg.pkl',
    'DecTree': 'model/DecTree.pkl',
    'KNeigh': 'model/KNeigh.pkl',
    'GausNB': 'model/GausNB.pkl',
    'RandFor': 'model/RandFor.pkl',
    'XGB': 'model/XGB.pkl'
}
classifiers_loaded = {key: joblib.load(path) for key, path in clf_paths.items()}

st.title('Digit Classifier Demo App')

# Upload eval data
upload_csv = st.file_uploader('Load CSV for Evaluation (needs "target" column)', type='csv')

if upload_csv is not None:
    eval_df = pd.read_csv(upload_csv)
    if 'target' not in eval_df.columns:
        st.error('Need "target" column with labels (0-9).')
    else:
        true_labels = eval_df['target']
        eval_features = eval_df.drop('target', axis=1)
        
        # Pick classifier
        chosen_clf = st.selectbox('Pick Classifier', list(classifiers_loaded.keys()))
        clf = classifiers_loaded[chosen_clf]
        
        # Predict
        preds = clf.predict(eval_features)
        probs = clf.predict_proba(eval_features)
        
        # Metrics
        acc = accuracy_score(true_labels, preds)
        auc = roc_auc_score(true_labels, probs, multi_class='ovr')
        prec = precision_score(true_labels, preds, average='macro')
        rec = recall_score(true_labels, preds, average='macro')
        f1 = f1_score(true_labels, preds, average='macro')
        mcc = matthews_corrcoef(true_labels, preds)
        
        st.subheader('Performance Metrics')
        perf_table = pd.DataFrame({
            'Measure': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
            'Score': [acc, auc, prec, rec, f1, mcc]
        })
        st.table(perf_table)
        
        # Heatmap for Confusion
        st.subheader('Confusion Heatmap')
        conf_mat = confusion_matrix(true_labels, preds)
        fig, axis = plt.subplots(figsize=(7, 5))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Greens', ax=axis,
                    xticklabels=range(10), yticklabels=range(10))
        axis.set_xlabel('Predicted Digit')
        axis.set_ylabel('True Digit')
        axis.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        # Report as table
        st.subheader('Class Report')
        rep_data = classification_report(true_labels, preds, output_dict=True, target_names=[str(i) for i in range(10)])
        rep_table = pd.DataFrame(rep_data).transpose().round(2)
        st.dataframe(rep_table.style.background_gradient(cmap='coolwarm'))