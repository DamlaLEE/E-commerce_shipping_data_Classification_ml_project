# E-commerce 'Order-He's Shipping Data Classification ML Project
### **Customer Rating Analysis and On-Time Delivery Prediction Using Order-He’s Shipping Data**

> 👉 [한국어 버전 README 보러가기 (View README in Korean)](README.ko.md)
---

## 1. Project Overview

This project is based on a fictional e-commerce service called **Order-He**, and aims to solve the following two key problems:

1. Analyze the factors that influence customer ratings and suggest ways to improve customer satisfaction.
2. Build a machine learning model to predict whether a product will be delivered on time.

> **Objective**: Design a data-driven system to enhance customer experience and improve operational efficiency.

---

## 2. Libraries Used

- **Basic**: `os`, `pickle`
- **Data Analysis**: `pandas`, `numpy`, `scipy`, `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`, `missingno`
- **Machine Learning Models**: `xgboost`, `lightgbm`

---

## 3. Data Description and Preprocessing

- **Data Source**: [Kaggle - Customer Analytics](https://www.kaggle.com/datasets/prachi13/customer-analytics)
- **Dataset**: 11 features + 1 target variable

| Column Name | Description |
|-------------|-------------|
| ID | Customer identifier |
| Warehouse_block | Storage warehouse section (A~F) |
| Mode_of_Shipment | Shipping method (Ship, Flight, Road) |
| Customer_care_calls | Number of customer service calls |
| Customer_rating | Customer rating (1 to 5) |
| Cost_of_the_Product | Product price |
| Prior_purchases | Number of previous purchases |
| Product_importance | Product importance level (low, medium, high) |
| Gender | Customer gender (M/F) |
| Discount_offered | Discount percentage |
| Weight_in_gms | Product weight (in grams) |
| Reached.on.Time_Y.N | On-time delivery status (0 = On time, 1 = Delayed) |

### 📌 Preprocessing Steps

- **Label Encoding**: Gender, Product Importance, Reached.on.Time_Y.N
- **One-Hot Encoding**: Mode_of_Shipment, Warehouse_block
- **Target Reset**: Reached.on.Time_Y.N → 1: On Time, 0: Delayed

---

## 4. Analysis 1: Factors Influencing Customer Ratings

- Correlation analysis between customer ratings and other features was conducted.
- Meaningful relationships identified:

  - **Positive correlation**: Prior_purchases, Customer_care_calls
  - **Negative correlation**: Reached.on.Time_Y.N, Discount_offered

> 📌 However, all correlation values were weak, indicating that no feature clearly determines customer ratings.

> 👉 Conclusion: The dataset is not suitable for building a reliable customer rating prediction model. Additional data is required.

---

## 5. Analysis 2: On-Time Delivery Prediction Model

### 🛠️ Modeling Process

1. **Target variable**: Reached.on.Time_Y.N (0 = Delayed, 1 = On Time)
2. **Models Used**:
   - Logistic Regression
   - SVM
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - LightGBM
3. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1-score
4. **Cross-Validation**: Stratified K-Fold
5. **Hyperparameter Tuning**: Randomized Search → Grid Search

### ✅ Model Performance Summary

- **Champion Model**: LightGBM
- **Best Parameters**:
  ```python
  {'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 3,
   'min_child_samples': 10, 'n_estimators': 400, 'subsample': 1.0}
  ```
- **📈 Evaluation Metrics**
    - **Accuracy**: 68%  
    - **Recall**: 95%  
    - **Precision**: 56%

> **Interpretation**  
    - The model accurately identifies on-time deliveries (low FN),  
    - making it effective for building customer trust.

---

### 📊 Confusion Matrix

| Actual / Predicted   | Predicted: Delayed (0) | Predicted: On Time (1) |
|----------------------|------------------------|--------------------------|
| **Actual: Delayed (0)** | TN (True Negative): 1013 | FP (False Positive): 975 |
| **Actual: On Time (1)** | FN (False Negative): 65   | TP (True Positive): 1247 |

- **TP (Correctly predicted on-time)**: 1247  
- **TN (Correctly predicted delayed)**: 1013  
- **FP (Incorrectly predicted on-time)**: 975  
- **FN (Incorrectly predicted delayed)**: 65

> ✅ **Interpretation**
> - **FP**: Customers may receive products sooner than expected — not critical  
> - **FN**: On-time deliveries predicted as delayed — could lead to dissatisfaction  
> → A model with **low FN is preferable**, making this model practical for real-world use.

---

### ⭐ Top 10 Most Important Features

![Top10_features](images/top10_important_feature_in_LGBM_model.png)

---

## 6. Practical Applications

### 🧪 Customer Rating Analysis
- Due to weak correlations, the current dataset is **not suitable for customer rating prediction**.
- → Future analyses will require **more comprehensive and realistic data**.

### 🚚 On-Time Delivery Prediction
- Integrate the model into the company’s website to **display the predicted delivery accuracy at checkout**.
- → This enhances **transparency**, builds **customer trust**, and can lead to **higher conversion rates**.

---

## 7. Limitations

- The dataset was sourced from **Kaggle** and may **not reflect real-world business conditions**.
- Some features might **lack practical meaning** or be randomly generated.
- For production use, it is essential to **retrain the model using actual internal business data**.

---

## 8. Appendix

- **Author**: DS_Yujin LEE  
- **Project Period**: 2025-04-11 to 2025-04-24  
- **Version**: v1.3  
- **Data Source**: [Kaggle - Customer Analytics](https://www.kaggle.com/datasets/prachi13/customer-analytics)
- **Updated date**: 2025-05-23

---
