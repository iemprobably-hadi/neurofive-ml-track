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

### 4. Advanced Evaluation & Hyperparameter Tuning (Task 5)
* **Metric Shift:** Beyond accuracy, evaluated model performance using **Precision**, **Recall**, and **F1-Score** to handle target class imbalance effectively.
* **Hyperparameter Optimization:** Used 5-fold cross-validated `GridSearchCV` across `C` regularization values and `solver` algorithms.
* **Performance Comparison:**

| Metric | Baseline Model | Tuned Model |
| :--- | :--- | :--- |
| **Accuracy** | 0.8101 | **0.8156** |
| **Precision** | 0.7857 | **0.8030** |
| **Recall** | 0.7432 | **0.7162** |
| **F1-Score** | 0.7639 | **0.7571** |

### 5. Customer Churn Prediction & Decision Trees (Task 6)
* **Dataset:** Telco Customer Churn dataset (7,043 customer accounts).
* **Class Imbalance:** Noted target distribution (~73% retained vs ~27% churned).
* **Models Trained:** Decision Tree Classifier (`max_depth=5`) vs. Logistic Regression.
* **Top 3 Drivers of Churn:**
  1. `Contract_Month-to-month`: High likelihood of immediate churn.
  2. `tenure`: Newer customers are significantly more prone to leaving.
  3. `TotalCharges` / `MonthlyCharges`: Price sensitivity impacts churn rate.

  ### 6. Production-Grade Pipelines & Feature Engineering (Task 7)
* **Feature Engineering:** Added `FamilySize` (`SibSp` + `Parch` + 1) and `IsAlone` binary indicator to capture family grouping survival dynamics.
* **Modular Architecture:** Used `ColumnTransformer` (`StandardScaler` for continuous numerical features, `OneHotEncoder` for categorical variables) combined with `LogisticRegression` into a unified `Pipeline`.
* **Model Serialization:** Exported the complete trained pipeline via `joblib` as `titanic_pipeline.joblib` for deployment readiness.