# Credit_Risk_Resampling

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. This program uses various techniques to train and evaluate models with imbalanced classes. It uses a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

---

## Technologies

This program runs on [Python 3.7](https://www.python.org/) and utilizes:
* [Jupyter Lab](https://jupyter.org/install)
* [Pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [imblearn](https://imbalanced-learn.org/stable/)

---
## Installation Guide

The dependencies needed to run the 'Credit-Risk-Resampling' program:

```python
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.metrics import classification_report_imbalanced

import warnings
warnings.filterwarnings('ignore')
```

---
## Usage

Launch Jupyter Lab with the file `credit_risk_resampling.ipynb`. Once the program is launched and the user moves through the program they will start seeing the results to the Machine Learning Models:
### Machine Learning Model 1:
![screen shot of ML model 1](https://user-images.githubusercontent.com/89755088/146666398-5af24c4c-6cd0-4d85-9b5d-ac97abfcf6e2.png)
### Machine Learning Model 2:
![screen shot of ML model 2](https://user-images.githubusercontent.com/89755088/146666415-90ee098f-73ab-4c78-9c95-091c75b12f9c.png)

---

## Overview of the Analysis

This program uses two machine learning techniques to analyze credit risk for lending activity from a peer-to-peer lending service company. The program uses a dataset that has a small amount of high risk loans(2500) compared to the amount of healthy loans(75036). When there is an imbalance in value counts as we find within this dataset; it is possible for the model to not deliver the best results when trying to identify the high risk loans. Since we find an imbalance in this instance, two machine learning techniques are used and compared to find which model did a better job at identifying high risk loans.

The machine learning models in this program use Logistic Regression on randomly split versions of the original data, using training sets and testing sets. The training sets are used to train the machine learning model. Once the model is trained it is then applied to the testing set to see how well it preforms identifying high risk loans. The results are given in terms of percentage of high risk loans correctly indentified versus percentage of high risk loans incorrectly identified. Since the provided dataset has a heavy imbalance in value counts for high risk and healthy loans, the second model utilizes an oversampleing technique that brings the value counts into balance, 56271 for high risk loans and 56271 for healthy loans. The results are then compared to the first model.


## Results

The two models are compared on three main metrics: accuracy, percision and recall. Accuracy - Is the percentage of correct predictions, which includes the target and features, or high risk loans and healthy loans. Precision - Is the percentage of correct predictions for the target, or high risk loans. It gives a good way to measure the percentage of false positives. Recall - Is the percentage of accurately identified targets, or high risk loans. It gives a good way to measure the percentage of targets that were missed, or false negatives.

* Machine Learning Model 1: Using Logistic Regression
    * Accuracy: 95.2%
    * Precision: 85%
    * Recall: 91%
    
* Machine Learning Model 2: Using Logistic Regression with Resampled(OverSampled) Value Counts
    * Accuracy: 99.4%
    * Precision: 84%
    * Recall: 99%
    
## Summary

Given the results, we can see that Model 2, which used the resampled(OverSampled) data, preformed much better than Model 1. Even though Model 2 saw a 1% decrease in precision, or in other words, saw a 1% increase in false positives. Model 2 with the resampled data, correctly identified 99% of high risk loans versus the first model which only correctly identified 91%. Model 2 would be my recommendation for which one the lending company should use.

---
## Contributors

Code for program functionality was done by Thomas Leahy, thomasleahy6@gmail.com

Starter Code provided by © 2020 - 2021 Trilogy Education Services, a 2U, Inc. brand.

---
## Licence
MIT Licence

2021 Thomas Leahy