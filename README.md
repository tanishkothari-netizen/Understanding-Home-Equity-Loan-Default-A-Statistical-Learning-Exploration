# Understanding Home Equity Loan Default: A Statistical Learning Exploration

A statistical learning study of **home-equity loan default prediction** using the HMEQ dataset. The project focuses on how missing-data handling, feature engineering, model selection, class imbalance, and decision thresholds affect predictive performance and practical lending decisions.

## Project Overview

The analysis uses **5,960 home-equity loan applications** with `BAD` as the binary target indicating loan default.

The project investigates several challenges commonly encountered in structured financial data:

- Missing values and informative missingness
- Class imbalance
- Skewed numerical variables
- Multicollinearity
- Feature engineering
- Model selection and hyperparameter tuning
- Classification threshold selection under asymmetric costs
- Model interpretation

The complete analysis is available in the accompanying Jupyter notebook:

[`Understanding Home Equity Loan Default: A Statistical Learning Exploration.ipynb`](./Understanding%20Home%20Equity%20Loan%20Default%3A%20A%20Statistical%20Learning%20Exploration.ipynb)

---

## Dataset

The HMEQ dataset contains **5,960 observations** of home-equity loan applications.

The target variable is:

- `BAD` — indicates whether the applicant defaulted on the loan.

The dataset contains both numerical and categorical information related to borrower characteristics, credit history, debt, loan amount, and property-related variables.

Notable data issues explored in the project include:

- Approximately **20% positive/default observations**
- Substantial missingness in some predictors
- Around **21% missingness in one variable**
- Strong correlation between `MORTDUE` and `VALUE` (approximately **0.88**)
- Skewed financial predictors
- Potential multicollinearity among credit and debt-related variables

---

## Analytical Workflow

The analysis follows a structured statistical learning pipeline:

```text
Data Exploration
      ↓
Missingness & Distribution Analysis
      ↓
Preprocessing
      ↓
Missing-Value Strategies
      ↓
Feature Engineering
      ↓
Model Comparison
      ↓
Hyperparameter Tuning
      ↓
Ablation Analysis
      ↓
Model Interpretation
      ↓
Threshold & Cost Analysis
```

### 1. Data Exploration

The initial analysis examines:

- Variable types and distributions
- Target-class balance
- Missing-value patterns
- Skewness
- Correlation structure
- Potential sources of multicollinearity

### 2. Missing-Data Analysis

Several approaches were compared:

- Dropping observations containing missing values
- Mean imputation
- Median imputation
- KNN imputation
- Imputation with explicit missingness indicators

A key result was that **missingness indicators carried substantial predictive information**.

For example:

| Missing-data strategy | ROC-AUC |
|---|---:|
| Mean imputation + missingness indicators | 0.9035 |
| Median imputation + missingness indicators | 0.9035 |
| KNN imputation + missingness indicators | 0.9022 |
| Median imputation, no indicators | 0.8011 |
| Mean imputation, no indicators | 0.7924 |
| Drop rows with missing values | 0.7858 |
| KNN imputation, no indicators | 0.7723 |

This motivates treating missingness itself as potentially informative rather than viewing it only as a nuisance to be removed.

---

## Models Compared

The project compares a range of statistical learning methods:

- Logistic Regression
- Ridge Regression
- LASSO
- Elastic Net
- K-Nearest Neighbors
- SVM with RBF kernel
- Decision Tree
- Random Forest
- Gradient Boosting

### 5-Fold Cross-Validation

Selected model-comparison results:

| Model | Mean ROC-AUC |
|---|---:|
| SVM (RBF) | **0.9311** |
| Gradient Boosting | **0.9307** |
| Random Forest | **0.9280** |
| LASSO | 0.9081 |
| Elastic Net | 0.9079 |
| Ridge | 0.9075 |
| Logistic Regression | 0.9069 |
| KNN | 0.9057 |
| Decision Tree | 0.8573 |

Hyperparameter tuning further improved the Random Forest model, reaching a cross-validated ROC-AUC of approximately **0.9326**.

---

## Ablation Study

A cumulative ablation experiment was used to measure how individual preprocessing and modeling decisions affected performance.

| Pipeline stage | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| Drop missing observations + Logistic Regression | 0.7587 | 0.4134 | 0.3958 |
| + Median imputation | 0.7719 | 0.5672 | 0.4403 |
| + Missingness indicators | 0.9020 | 0.7851 | 0.6680 |
| + Feature engineering | 0.9121 | 0.7990 | 0.6923 |
| + Tuned regularization | 0.9123 | 0.7999 | 0.6873 |
| Switch to Random Forest | 0.9319 | 0.8281 | 0.6774 |
| Final tuned model | **0.9430** | **0.8491** | **0.7318** |

The ablation study demonstrates that performance gains came not from a single modeling choice, but from the combination of **missingness-aware preprocessing, feature engineering, model selection, and tuning**.

---

## Feature Engineering

Feature engineering was used to incorporate domain-relevant relationships that were not directly represented by the raw variables.

One example is:

\[
\text{CREDIT\_STABILITY\_INDEX}
=
\frac{\text{CLAGE}}{\text{NINQ}+1}
\]

This captures credit-history age relative to the frequency of recent credit inquiries.

The analysis also examines variables related to delinquency, derogatory credit history, debt, and repayment burden.

---

## Model Interpretation

The final analysis goes beyond predictive performance and examines **why the model makes its predictions**.

Interpretation techniques include:

- Permutation importance
- SHAP analysis
- Partial dependence analysis
- Examination of engineered features
- Threshold-based decision analysis

The analysis highlights the predictive role of credit-history, delinquency/derogatory-history, debt-to-income, and missingness-related information.

---

## Threshold Analysis

A classification threshold of `0.5` is not necessarily optimal in a lending problem because false negatives and false positives can have different economic consequences.

The project therefore evaluates multiple operating thresholds.

| Threshold | Precision | Recall | F1 | False Negatives | False Positives | Expected Cost |
|---|---:|---:|---:|---:|---:|---:|
| 0.50 (default) | 0.8364 | 0.6027 | 0.7006 | 118 | 35 | 625 |
| 0.33 (F1-optimal) | 0.7468 | 0.7744 | **0.7603** | 67 | 78 | 413 |
| 0.14 (5:1 cost-optimal) | 0.5456 | **0.9259** | 0.6866 | **22** | 229 | **339** |

This illustrates an important practical point:

> The best classification threshold depends on the objective function, not only on predictive accuracy.

When the cost of missing a default is high, a lower threshold can substantially increase recall and reduce expected decision cost.

---

## Key Findings

### 1. Missingness can be predictive

Adding missingness indicators produced a large improvement in ROC-AUC compared with simple imputation without indicators.

### 2. Model choice matters

Nonlinear models such as SVM, Gradient Boosting, and Random Forest substantially outperformed basic linear models on this dataset.

### 3. Feature engineering adds signal

Domain-inspired variables can capture relationships that are difficult for a model to infer from raw predictors alone.

### 4. ROC-AUC is not the whole story

Because the target is imbalanced and the costs of errors are asymmetric, **PR-AUC, F1, recall, and expected cost** provide additional information beyond ROC-AUC.

### 5. Decision thresholds should reflect the application

A threshold selected for maximum F1 can differ substantially from one selected to minimize a business-specific cost function.

---

## Technologies

**Language**

- Python

**Core Libraries**

- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib
- Seaborn
- SHAP

**Environment**

- Jupyter Notebook

---

## Repository Structure

```text
.
├── Understanding Home Equity Loan Default:
│   A Statistical Learning Exploration.ipynb
└── README.md
```

---

## Project Focus

This project was designed as a practical study of **statistical learning for credit-risk modeling**, with particular emphasis on:

```text
Missing Data
→ Feature Engineering
→ Model Selection
→ Validation
→ Interpretation
→ Cost-Sensitive Decision Making
```

Rather than treating preprocessing as a purely mechanical step, the analysis evaluates how statistical assumptions and modeling choices influence the final predictive and decision-making performance.

---

## Author

**Tanish Kothari**

Bachelor of Statistical Data Science  
Indian Statistical Institute, Delhi

GitHub: [@tanishkothari-netizen](https://github.com/tanishkothari-netizen)
