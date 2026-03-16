import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                              recall_score, f1_score, roc_auc_score, RocCurveDisplay)
from sklearn.ensemble import RandomForestClassifier

# Model training heavily based off of the following article
# https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/

df = pd.read_csv('Data/compas-scores-two-years.csv')

cols_to_drop = [
    'id','name','first','last','dob','compas_screening_date','c_case_number',
    'c_offense_date','c_arrest_date','c_days_from_compas','decile_score',
    'score_text','is_recid','is_violent_recid','event','v_type_of_assessment',
    'v_decile_score','v_score_text','v_screening_date','in_custody','out_custody',
    'start','end','c_jail_in','c_jail_out','r_case_number','r_charge_degree',
    'r_days_from_arrest','r_offense_date','r_charge_desc','r_jail_in','r_jail_out',
    'vr_case_number','vr_charge_desc','vr_offense_date','screening_date',
    'violent_recid','decile_score.1','priors_count.1','days_b_screening_arrest',
    'vr_charge_degree','age_cat','type_of_assessment'
]

filtered_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
demographics = filtered_df[['race', 'sex', 'age']].copy()

encoders = {}
df_encoded = filtered_df.copy()
for col in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le

feature_cols = ['sex', 'age', 'race', 'juv_fel_count', 'juv_misd_count',
                'juv_other_count', 'priors_count', 'c_charge_degree', 'c_charge_desc']

feature_cols_no_demo = ['juv_fel_count', 'juv_misd_count', 'juv_other_count',
                        'priors_count', 'c_charge_degree', 'c_charge_desc']

label = 'two_year_recid'

X_train_full, X_test_full, y_train, y_test = train_test_split(
    df_encoded[feature_cols], df_encoded[label], test_size=0.2, random_state=42
)

X_train_nodemo, X_test_nodemo, _, _ = train_test_split(
    df_encoded[feature_cols_no_demo], df_encoded[label], test_size=0.2, random_state=42
)

demo_test = demographics.loc[X_test_full.index].reset_index(drop=True)
X_test_full = X_test_full.reset_index(drop=True)
X_test_nodemo = X_test_nodemo.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_nodemo = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_train_full, y_train)
rf_nodemo.fit(X_train_nodemo, y_train)

# Metrics
for name, model, X_test in [('With Demographics', rf_full, X_test_full), ('Without Demographics', rf_nodemo, X_test_nodemo)]:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(name)
    print(f"Accuracy:  {accuracy_score(y_test, preds):.3f}")
    print(f"Precision: {precision_score(y_test, preds):.3f}")
    print(f"Recall:    {recall_score(y_test, preds):.3f}")
    print(f"F1:        {f1_score(y_test, preds):.3f}")

# Plotting Confusion Matrix
conf_matrix = confusion_matrix(y_test, preds)
class_names = ['No Recid', 'Recid']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, 
            xticklabels=class_names, yticklabels=class_names)

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()

# Feature importances
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (model, cols, title) in zip(axes, [
    (rf_full,   feature_cols,         'With Demographics'),
    (rf_nodemo, feature_cols_no_demo, 'Without Demographics')
]):
    imp_df = pd.DataFrame({'Feature': cols, 'Importance': model.feature_importances_})
    imp_df = imp_df.sort_values('Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=imp_df, palette='magma', ax=ax)
    ax.set_title(f'Feature Importances — {title}')

plt.tight_layout()
plt.savefig('feature_importances.png', dpi=150)
plt.show()

# Fairness analysis - written with the help of Gemini
def fairness_by_group(model, X_test, y_test, demographics, group_col):
    df = demographics[[group_col]].copy()
    df['true'] = y_test.values
    df['pred'] = model.predict(X_test)

    results = []
    for group_val, group_df in df.groupby(group_col):
        if len(group_df) < 30:
            continue
        cm = confusion_matrix(group_df['true'], group_df['pred'])
        tn, fp, fn, tp = cm.ravel()
        results.append({
            'Group':     group_val,
            'N':         len(group_df),
            'Accuracy':  (tp + tn) / (tp + tn + fp + fn),
            'FPR':       fp / (fp + tn) if (fp + tn) > 0 else np.nan,
            'FNR':       fn / (fn + tp) if (fn + tp) > 0 else np.nan,
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else np.nan,
            'Recall':    tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        })
    return pd.DataFrame(results).set_index('Group').round(3)

def plot_fairness(metrics_df, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, (metric, color) in zip(axes, [('FPR', 'steelblue'), ('FNR', 'coral')]):
        metrics_df[metric].plot(kind='bar', ax=ax, color=color, edgecolor='black')
        ax.set_title(f'{metric} by Group')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()

for col in ['race', 'sex']:
    for name, model, X_test in [('With Demographics', rf_full, X_test_full), ('Without Demographics', rf_nodemo, X_test_nodemo)]:
        result = fairness_by_group(model, X_test, y_test, demo_test, col)
        print(result)

# Demographic parity
demo_test['pred_full']   = rf_full.predict(X_test_full)
demo_test['pred_nodemo'] = rf_nodemo.predict(X_test_nodemo)

parity_full   = demo_test.groupby('race')['pred_full'].mean().round(3)
parity_nodemo = demo_test.groupby('race')['pred_nodemo'].mean().round(3)

parity_table = pd.DataFrame({
    'With Demographics':    parity_full,
    'Without Demographics': parity_nodemo
})
parity_table.index.name = 'Group'
print(parity_table)
