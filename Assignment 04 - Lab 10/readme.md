# Comparative Implementation of Machine Learning Regression Algorithms

**Name:** Muhammad Ibrahim Asghar  
**Registration No:** FA23-BAI-046  

---

## 1. Project Overview
This repository presents a comparative Python implementation of six supervised machine learning regression algorithms applied to the **California Housing Prices Dataset** (`camnugent`, Kaggle). All models are trained and evaluated on the exact same feature set, target variable (`median_house_value`), and **80:20 train-test split** (`random_state=42`) to ensure a fair comparison.

### Preprocessing & Experimental Setup
* **Missing Value Imputation:** Missing values in `total_bedrooms` were imputed using the column median.
* **Categorical Encoding:** The `ocean_proximity` feature was converted using one-hot encoding (`drop_first=True`), with special characters sanitized for XGBoost compatibility.
* **Feature Scaling:** `StandardScaler` was applied to the input features for scale-sensitive models (**Linear Regression**, **Neural Network**, and **KNN**). Target scaling was also applied during Neural Network training to ensure stable convergence.
* **Evaluation Metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$).
* **Visualizations:** Actual vs. Predicted scatter plots for all six models are generated and saved inside the accompanying Jupyter/Colab notebook (`.ipynb`).

---

## 2. Comparative Performance Table

| Rank | Algorithm | MAE ($) | MSE | RMSE ($) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **XGBoost Regression** | 32,430.71 | 2,349,141,133.69 | **48,467.94** | **0.8207** |
| **2** | **Random Forest Regression** | **31,639.37** | 2,404,745,975.12 | 49,038.21 | 0.8165 |
| **3** | **Neural Network Regression (MLP)** | 36,395.75 | 2,936,378,007.70 | 54,188.36 | 0.7759 |
| **4** | **K-Nearest Neighbors (KNN)** | 40,854.40 | 3,765,913,279.63 | 61,367.04 | 0.7126 |
| **5** | **Decision Tree Regression** | 44,180.85 | 4,865,868,836.96 | 69,755.78 | 0.6287 |
| **6** | **Linear Regression** | 50,670.74 | 4,908,476,721.16 | 70,060.52 | 0.6254 |

---

## 3. Strengths and Limitations of Each Algorithm

### 1. XGBoost Regression
* **Strengths:** Uses gradient boosting with built-in regularization to sequentially correct residual errors, making it highly effective at capturing complex non-linear patterns and geographic interactions (e.g., latitude/longitude vs. price) in tabular data.
* **Limitations:** More sensitive to hyperparameter tuning than Random Forest and slightly less interpretable as a "black-box" ensemble.

### 2. Random Forest Regression
* **Strengths:** Reduces the high variance of individual decision trees by averaging predictions across 100 trees (bagging). Handles non-linear relationships well, requires no feature scaling, and achieved the lowest MAE ($31,639.37).
* **Limitations:** Higher memory usage and slower inference compared to linear models; cannot extrapolate beyond the range of values seen in the training set.

### 3. Neural Network Regression (MLP)
* **Strengths:** Capable of modeling complex, smooth non-linear relationships across high-dimensional features once properly scaled.
* **Limitations:** Requires careful feature and target scaling, is computationally heavier to train, and generally underperforms gradient-boosted trees on medium-sized tabular datasets without extensive architecture tuning.

### 4. K-Nearest Neighbors (KNN) Regression
* **Strengths:** Simple, non-parametric, and requires virtually no training time; naturally captures local geographic neighborhoods when features are scaled.
* **Limitations:** Prediction time scales poorly with dataset size, is sensitive to irrelevant features and the choice of $k$, and struggles near the boundaries of the feature space.

### 5. Decision Tree Regression
* **Strengths:** Easy to interpret, fast to train, requires no feature scaling, and naturally models non-linear splits.
* **Limitations:** Highly prone to overfitting the training data, resulting in high variance and poor generalization on unseen test samples ($R^2 = 0.6287$).

### 6. Linear Regression
* **Strengths:** Fastest to train, fully interpretable coefficients, and serves as a strong baseline model.
* **Limitations:** Assumes a strictly linear relationship between features and house prices. It fails to capture non-linear geographic effects (such as coastal clusters), resulting in the highest error across all metrics.

---

## 4. Conclusion & Best-Performing Model

**Best-Performing Algorithm:** **XGBoost Regression**

**Justification:**  
Based on the experimental results, **XGBoost Regression** is the best-performing model overall, achieving the highest **$R^2$ Score (0.8207)** and the lowest **MSE (2.35 × 10⁹)** and **RMSE ($48,467.94)**. This means XGBoost explains approximately **82.07%** of the variance in California housing prices while minimizing large prediction errors better than any other model.

**Random Forest Regression** is a very close runner-up ($R^2 = 0.8165$, RMSE = $49,038.21) and slightly outperforms XGBoost on **MAE ($31,639.37 vs. $32,430.71)**. Because RMSE penalizes large outliers more heavily than MAE, this indicates that while Random Forest has a marginally lower average error on typical homes, XGBoost is more reliable at avoiding severe mispredictions on high-value or outlier properties. Both tree-based ensembles significantly outperform Linear Regression ($R^2 = 0.6254$) and single Decision Trees ($R^2 = 0.6287$), confirming that housing prices depend heavily on non-linear feature interactions that ensemble methods capture best.