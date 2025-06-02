# House-Price-Prediction-using-Machine-Learning

This project demonstrates a complete machine learning workflow to predict housing prices in Boston using the XGBoost Regressor. It includes essential steps such as data pre-processing, analysis, model training, and evaluation.

## 📊 Objective

The goal is to predict house prices in the Boston area using structured housing data. This involves building a regression model that accurately estimates the prices based on various features like crime rate, number of rooms, and accessibility to highways.

---

## 🛠️ Workflow

### 1. House Price Data

The project utilizes the Boston Housing Dataset, a classic dataset in regression analysis. It contains 506 entries and 13 numerical/categorical features plus the target variable (housing price).

Key features include:

* CRIM: Crime rate
* RM: Average number of rooms
* LSTAT: % lower status of the population
* PTRATIO: Pupil-teacher ratio
* Others relating to property, location, and economy

> Note: The dataset is loaded via sklearn or fetched from external sources if deprecated.

---

Dataset: <a href="https://github.com/Shibaditya00/House-Price-Prediction-using-Machine-Learning/tree/main">Boston House Price Dataset</a>

---

### 2. Data Pre-processing

To ensure model compatibility and robustness:

* Null/missing values are checked (none found in this dataset)
* Data types and feature distributions are explored
* Features and target variable are separated into `X` and `Y`

---

### 3. Data Analysis

Exploratory Data Analysis (EDA) is performed using:

* Seaborn heatmaps to visualize feature correlation
* Highly correlated variables are identified to inform model performance and feature importance.

---

### 4. Train-Test Split

The dataset is divided:

* 80% for training
* 20% for testing

This ensures model generalization and unbiased performance assessment.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```

---

### 5. XGBoost Regressor

A powerful gradient boosting model is used:

* Handles complex patterns
* Robust to overfitting
* High accuracy in structured data

```python
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, y_train)
```

---

### 6. Evaluation

Performance is evaluated using:

* R² Score
* Mean Squared Error (MSE)

The model achieves a strong R² score, indicating high predictive performance on unseen data.

---

## 📌 Conclusion

This project illustrates a full-cycle machine learning regression pipeline from data loading to model evaluation. The use of XGBoost offers strong results, and the methodology is easily extensible to similar datasets.

---

## 🧰 Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* XGBoost

---

