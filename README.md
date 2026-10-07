 💳 Credit Card Default Prediction

 📌 Project Overview

This project focuses on predicting whether a customer will **default on their credit card payment** in the next month using Machine Learning.

The project includes **Exploratory Data Analysis (EDA), data preprocessing, categorical encoding, outlier treatment, class balancing, feature selection, feature transformation, and classification model building**.

The main objective is to understand customer financial information and build machine learning models that can predict credit card payment default.

---

 🎯 Objectives

- Perform Exploratory Data Analysis (EDA)
- Understand the structure and distribution of the dataset
- Handle missing values and duplicate records
- Detect and treat outliers
- Convert categorical variables into numerical form
- Handle class imbalance using SMOTE
- Select important features
- Transform skewed numerical features
- Standardize the selected features
- Build and compare multiple classification models
- Evaluate models using Accuracy, Precision, Recall and F1-Score

---

 📊 Dataset

The project uses a **Credit Card Default dataset** containing customer demographic, billing and payment information.

Important features include:

- `LIMIT_BAL` – Credit limit
- `SEX` – Gender
- `EDUCATION` – Education level
- `MARRIAGE` – Marital status
- `AGE` – Customer age
- `PAY_0` to `PAY_6` – Repayment status
- `BILL_AMT1` to `BILL_AMT6` – Bill amounts
- `PAY_AMT1` to `PAY_AMT6` – Previous payment amounts
- `default payment next month` – Target variable

The target variable was renamed as:

```text
target
```

---

 🔍 Exploratory Data Analysis

The following EDA techniques were performed:

- Dataset inspection using `head()` and `tail()`
- Dataset information using `info()`
- Dataset shape analysis
- Column identification
- Statistical summary using `describe()`
- Missing value checking
- Duplicate value checking
- Correlation analysis
- Correlation heatmap
- Boxplot analysis
- Target class distribution

---

🧹 Data Preprocessing

1. Column Renaming

The payment and billing columns were renamed to make them easier to understand.

For example:

```text
PAY_0 → SEP_PAY
PAY_2 → AUG_PAY
BILL_AMT1 → SEP_BILL
PAY_AMT1 → SEP_PAYMENT
```

The target column was renamed:

```text
default payment next month → target
```

2. Categorical Data Conversion

Categorical values were converted into meaningful labels.

Examples:

```text
SEX:
1 → M
2 → F
```

```text
MARRIAGE:
1 → MARRIED
2 → SINGLE
3 → OTHERS
```

```text
EDUCATION:
1 → UG
2 → PG
3 → HIGH_SCHOOL
4 → OTHERS
```

The target was converted into:

```text
1 → YES
0 → NO
```

---

 📦 Outlier Treatment

Outliers were detected using boxplots and treated using the **IQR (Interquartile Range) method**.

The IQR method uses:

```text
IQR = Q3 - Q1
```

Lower and upper limits were calculated as:

```text
Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

Values outside these limits were replaced with the corresponding boundary values.

---

 🔢 Encoding

Different encoding techniques were used for categorical variables.

# Label Encoding

Label Encoding was used for:

- `target`
- `SEX`

# One-Hot Encoding

One-Hot Encoding was used for:

- `EDUCATION`
- `MARRIAGE`

This converts categorical values into numerical columns that can be used by machine learning algorithms.

---

#⚖️ Handling Class Imbalance

The target classes were balanced using **SMOTE (Synthetic Minority Over-sampling Technique)**.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE()
x_smote, y_smote = smote.fit_resample(X, y)
```

SMOTE generates synthetic samples for the minority
