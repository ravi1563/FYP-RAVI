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
