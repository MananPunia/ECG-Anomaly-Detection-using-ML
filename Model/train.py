import pandas as pd
import numpy as np
import joblib
import json

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Loading data...")
train_df = pd.read_csv("Train&Test/mitbih_train.csv", header=None)
test_df = pd.read_csv("Train&Test/mitbih_test.csv", header=None)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_model.fit(X_train_bal, y_train_bal)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))

print("Training KNN...")
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_model.fit(X_train_bal, y_train_bal)
knn_acc = accuracy_score(y_test, knn_model.predict(X_test_scaled))

print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train_bal, y_train_bal)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test_scaled))

print("Saving models and results...")
joblib.dump(rf_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

results_dict = {
    "Random Forest": float(rf_acc),
    "KNN": float(knn_acc),
    "Logistic Regression": float(lr_acc)
}

with open("results.json", "w") as f:
    json.dump(results_dict, f)

print("✅ Training Complete and Assets Saved.")