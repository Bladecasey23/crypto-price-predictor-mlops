import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

df = pd.read_csv('btc_features_clean.csv', parse_dates=['timestamp'])

# Features we'll actually feed into the model
feature_cols = ['pct_change_1h', 'rolling_mean_6h', 'rolling_mean_24h', 'price_vs_24h_avg', 'volatility_6h']
X = df[feature_cols]
y = df['target']

# IMPORTANT: for time series, we do NOT shuffle when splitting.
# We train on the past and test on the future shuffling would let the model
# "see" future patterns during training, which is unrealistic and inflates accuracy.
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds))
print("Recall:", recall_score(y_test, preds))
print("F1:", f1_score(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

joblib.dump(model, 'btc_model.pkl')
print("Model saved to btc_model.pkl")