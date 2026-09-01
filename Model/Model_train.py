import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv('../Dataset/creditcard.csv')

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# 2. Features and Target separation
X = df.drop(columns=['Class'])
y = df['Class']

# 3. Feature Scaling (Time and Amount)
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# 4. Train-Test Split (Stratified to maintain class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train Model 1: Logistic Regression
print("\n--- Training Logistic Regression ---")
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1]))

# 6. Train Model 2: Random Forest
print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest ROC-AUC:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

# 7. Save Both Models & Scaler in a Single PKL File (Best Practice for Streamlit)
os.makedirs('saved_models', exist_ok=True)

# Option A: Single Dictionary File (સુવિધાજનક)
all_in_one_pkg = {
    'scaler': scaler,
    'logistic_model': lr_model,
    'random_forest_model': rf_model
}

with open('saved_models/fraud_detection_all_models.pkl', 'wb') as f:
    pickle.dump(all_in_one_pkg, f)

# Option B: Separate PKL Files
with open('saved_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('saved_models/logistic_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('saved_models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("\n Success! All models and scaler saved in 'saved_models/' directory.")