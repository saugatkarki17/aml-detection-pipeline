import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, roc_auc_score,
    average_precision_score, classification_report
)
from imblearn.over_sampling import SMOTE

df = pd.read_csv('./data/processed_transactions.csv')
print("Dataset loaded:", df.shape)

# Ensure essential columns exist and are numeric
REQUIRED_COLUMNS = [
    'amount_paid', 'amount_received', 'txn_count', 'total_sent',
    'hour_of_day', 'day_of_week', 'is_laundering'
]

missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Feature Engineering to limit the inputs
df['amount_diff'] = abs(df['amount_paid'] - df['amount_received'])
df['ratio_received_paid'] = df['amount_received'] / (df['amount_paid'] + 1e-5)
df['avg_txn_amount'] = df['total_sent'] / (df['txn_count'] + 1e-5)
df['z_score_paid'] = (df['amount_paid'] - df['amount_paid'].mean()) / df['amount_paid'].std()

# Preparing data for model training
X = df.drop(columns=['is_laundering'])
y = df['is_laundering']

smote = SMOTE(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = []
scores = []

print("Now starting 5-fold training...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"▶️ Fold {fold+1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=9,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train_res, y_train_res)

    y_proba = model.predict_proba(X_val)[:, 1]
    roc_score = roc_auc_score(y_val, y_proba)
    scores.append(roc_score)
    print(f"Fold {fold+1} ROC AUC: {roc_score:.4f}")

    models.append(model)

# Select best Model
best_model_idx = np.argmax(scores)
best_model = models[best_model_idx]
manual_threshold = 0.97  # Manual entry of threshold(F1 Based because of low flagging rate)

print(f"\nBest Fold: {best_model_idx + 1}")
print(f"ROC AUC: {scores[best_model_idx]:.4f}")
print(f"Using Manual Threshold: {manual_threshold:.3f}")

# Here save model
joblib.dump({
    "model": best_model,
    "threshold": manual_threshold,
    "features": X.columns.tolist()
}, "xgb_AML_model_advanced.joblib")

print("Model saved to: xgb_AML_model_advanced.joblib")
