# Neurofive Machine Learning Track: Titanic Survival Prediction

This repository contains my step-by-step progress through the Neurofive Solutions Machine Learning Track using the Titanic dataset.

## 🛠️ Project Workflow

### 1. Exploratory Data Analysis (Task 1)
* Inspected dataset structure (891 rows, 12 columns).
* Identified missing values in `Age` (~20%), `Cabin` (~77%), and `Embarked` (<1%).

### 2. Data Cleaning & Visualization (Task 2)
* Imputed `Age` with median values and `Embarked` with mode ('S').
* Dropped `Cabin` due to excessive missingness.
* Built Seaborn visualizations showing strong survival correlation with `Sex` (female survival rate ~74% vs. male ~19%) and `Pclass`.

### 3. Machine Learning Classification (Task 3)
* **Encoding:** Applied One-Hot Encoding (`pd.get_dummies`) for `Sex` and `Embarked`.
* **Data Split:** 80% Training / 20% Testing split using `train_test_split`.
* **Algorithm:** Logistic Regression (`scikit-learn`).
* **Evaluation:** Evaluated model using `accuracy_score` and `confusion_matrix`.

---

## 📊 Final Model Accuracy
* **Accuracy Score:** **~80%** on unseen test data.