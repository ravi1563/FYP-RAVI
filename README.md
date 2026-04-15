# FYP-RAVI

# Cardiovascular Disease Risk Prediction Using Machine Learning

The main aim of this work is to explore how machine learning can be used to predict the risk of cardiovascular disease (CVD) using patient clinical data.

---

## About the Project
Cardiovascular disease is one of the major health issues worldwide, and early prediction can help improve treatment decisions.

In this project, I compared different machine learning models to find which one performs best for predicting heart disease risk.

I also used explainable AI techniques to understand which medical features have the biggest impact on predictions.

---

## Dataset
The dataset used for this project was taken from **Mendeley Data** and contains **1,319 patient records**.

Some of the main features include:

- Age
- Gender
- Heart Rate
- Blood Pressure
- Blood Sugar
- CK-MB
- Troponin

The target variable indicates whether the patient is **positive or negative for cardiovascular disease**.

---

## Models Applied
The following models were tested and compared:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

---

## Approach
The workflow included:

- data preprocessing
- feature scaling
- train-test split
- hyperparameter tuning using Optuna
- cross-validation
- SHAP explainability
- testing SMOTE and ADASYN for class balancing

---

## Final Results
Among all models, **Random Forest performed the best**.

- **Accuracy:** 98.48%
- **F1 Score:** 0.9848
- **ROC-AUC:** 0.994

The most important predictive features were:

- Troponin
- CK-MB
- Age

---

## Key Outcome
The results showed that Random Forest gives highly accurate predictions for CVD risk.

The SHAP analysis also helped identify the most important clinical biomarkers influencing the model decisions.

Interestingly, applying balancing techniques such as SMOTE and ADASYN did not improve the model performance.

---

## Tools Used
Python, Pandas, NumPy, Scikit-learn, Optuna, SHAP, Matplotlib

---

## Author
**Ravi Kiran Deevi**  
MSc Data Science  
University of Hertfordshire


# Code Workflow

## 1. Import Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 2. Load Dataset
```python
df = pd.read_csv("Medicaldataset.csv")
df.head()
```

---

## 3. Exploratory Data Analysis (EDA)
### Dataset Shape and Columns
```python
df.shape
df.columns
```

### Missing Values
```python
df.isnull().sum()
```

### Statistical Summary
```python
df.describe()
```

### Target Distribution
```python
sns.countplot(x='Result', data=df)
plt.title("Class Distribution")
plt.show()
```

### Feature Distribution
```python
df.hist(figsize=(12,10))
plt.tight_layout()
plt.show()
```

### Correlation Heatmap
```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()
```

---

## 4. Data Preprocessing
```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df = df.dropna()

X = df.drop("Result", axis=1)
y = df["Result"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

---

## 5. Model Training
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

lr = LogisticRegression()
svm = SVC(probability=True)
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
rf.fit(X_train, y_train)
```

---

## 6. Model Evaluation
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

pred_rf = rf.predict(X_test)

accuracy_score(y_test, pred_rf)
precision_score(y_test, pred_rf)
recall_score(y_test, pred_rf)
f1_score(y_test, pred_rf)
```

---

## 7. Hyperparameter Tuning (Optuna)
```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return f1_score(y_test, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
```

---

## 8. Cross Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf, X_scaled, y, cv=5)
print(scores.mean())
print(scores.std())
```
