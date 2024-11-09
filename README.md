# Credit Card Fraud Detection Project

## Overview
This project involves building a machine learning model to detect fraudulent credit card transactions. The dataset used is from a large set of credit card transactions with over 1.2 million entries.

## Dataset Information
The dataset contains 23 columns, including:
- **Transaction Date and Time**
- **Credit Card Number (cc_num)**
- **Merchant Information (merchant)**
- **Transaction Category (category)**
- **Amount (amt)**
- **Customer Information (first, last, gender, etc.)**
- **Location Data (city, state, zip, lat, long)**
- **Transaction Outcome (is_fraud)**

### Data Preprocessing
To prepare the data for modeling, the following preprocessing steps were applied:
1. Dropped non-essential columns such as `Unnamed: 0`, `trans_num`, and `street`.
2. Converted date and time fields into appropriate formats.
3. Checked for null values and ensured data consistency.

## Libraries Used
- **Numpy**: For numerical operations
- **Pandas**: For data manipulation and analysis
- **Seaborn & Matplotlib**: For data visualization
- **Scikit-learn**: For model building and evaluation

## Model Development
### Algorithms Used:
- **Logistic Regression**: A baseline model for binary classification.
- **Support Vector Classifier (SVC)**: To explore non-linear relationships and improve accuracy.

### Model Evaluation Metrics:
- **Confusion Matrix**
- **Accuracy Score**

## Code Snippets
### Importing Required Libraries:
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
