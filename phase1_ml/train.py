import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# Load Dataset
# =========================

data = pd.read_csv("../dataset/sonar.csv", header=None)

# Features and Labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert Labels
y = y.map({'R': 0, 'M': 1})

# =========================
# Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Feature Scaling
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# SVM Model
# =========================

svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True
)

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)

print(f"\nSVM Accuracy: {svm_accuracy * 100:.2f}%")

# =========================
# Random Forest
# =========================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# =========================
# Ensemble Learning
# =========================

lr_model = LogisticRegression()

knn_model = KNeighborsClassifier()

ensemble_model = VotingClassifier(
    estimators=[
        ('svm', svm_model),
        ('rf', rf_model),
        ('lr', lr_model),
        ('knn', knn_model)
    ],
    voting='soft'
)

ensemble_model.fit(X_train, y_train)

ensemble_pred = ensemble_model.predict(X_test)

ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

print(f"Ensemble Accuracy: {ensemble_accuracy * 100:.2f}%")

# =========================
# Evaluation
# =========================

print("\nClassification Report:")
print(classification_report(y_test, ensemble_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, ensemble_pred))


joblib.dump(svm_model, "../models/svm_model.pkl")

print("\nSVM model saved successfully.")

joblib.dump(scaler, "../models/scaler.pkl")

print("Scaler saved successfully.")

models = ['SVM', 'Random Forest', 'Ensemble']

accuracies = [
    svm_accuracy * 100,
    rf_accuracy * 100,
    ensemble_accuracy * 100
]

plt.figure(figsize=(8,5))

plt.bar(models, accuracies)

plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")

plt.ylim(0, 100)

plt.show()