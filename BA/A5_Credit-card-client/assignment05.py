# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 23:03:31 2017

@author: Hanbin Seo
"""

import numpy as np
import pandas as pd
f_dir = "D:/course_2017_spring/BA/Data/default-of-credit-card-clients-dataset/"
data =pd.read_csv(f_dir+"UCI_Credit_Card.csv")
copy_data = data.copy()

TARGET_COL = 'default.payment.next.month'
print(data[TARGET_COL].value_counts())

##################################
### Data Explore & Pre-Processing
##################################

rm_colnames = ['ID', 'SEX', 'EDUCATION', 'MARRIAGE']
pay_colnames = ['PAY_{}'.format(i) for i in range(7) if i!=1]
data = data.drop(rm_colnames, axis=1)
data = data.drop(pay_colnames, axis=1)

"""
ID변수와 범주형 변수 제거
"""

##################################
### Input/Output(label) data split
##################################

colnames = [col for col in data.columns.values if col != TARGET_COL]
X = data[colnames]
y = data[TARGET_COL]


##################################
### Function of Logistic Regression 
##################################

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def LogisticReg(X, y) :
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    skf = StratifiedKFold(n_splits=5)
    idx_train, idx_test = [], []
    for train, test in skf.split(X, y) :
        idx_train.append(train)
        idx_test.append(test)

    clf = LogisticRegression()
    acc_li, prec_li, rec_li, f1_li = [], [], [], []
    for k in range(len(idx_train)) :
        X_train, y_train = X.iloc[idx_train[k]].values, y.iloc[idx_train[k]].values
        X_test, y_test = X.iloc[idx_test[k]].values, y.iloc[idx_test[k]].values
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    
        acc_li.append(metrics.accuracy_score(y_test, y_pred))
        prec_li.append(metrics.precision_score(y_test, y_pred))
        rec_li.append(metrics.recall_score(y_test, y_pred))
        f1_li.append(metrics.f1_score(y_test, y_pred))
    
    res_acc = np.array(acc_li).mean()
    res_prec = np.array(prec_li).mean()
    res_rec = np.array(rec_li).mean()
    res_f1 = np.array(f1_li).mean()
    return res_acc, res_prec, res_rec, res_f1


##################################
### [1] Non Sampling
##################################

result_dic = {"Non Sampling" : LogisticReg(X,y)}


##################################
### [2] Over Sampling
##################################

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_sample(X,y)
result_dic["Over Sampling"] = LogisticReg(X_resampled, y_resampled)


##################################
### [3] Under Sampling
##################################

from imblearn.under_sampling import RandomUnderSampler
ros = RandomUnderSampler()
X_resampled, y_resampled = ros.fit_sample(X,y)
result_dic["Under Sampling"] = LogisticReg(X_resampled, y_resampled)


##################################
### [4] SMOTE
##################################

from imblearn.over_sampling import SMOTE
sm = SMOTE(kind='regular')
X_resampled, y_resampled = sm.fit_sample(X,y)
result_dic["SMOTE"] = LogisticReg(X_resampled, y_resampled)


##################################
### [5] Tomek Links
##################################

from imblearn.under_sampling import TomekLinks
tl = TomekLinks(return_indices=True)
X_resampled, y_resampled, idx_resampled = tl.fit_sample(X,y)
idx_removed = np.setdiff1d(np.arange(X_resampled.shape[0]), idx_resampled)
result_dic["Tomek Links"] = LogisticReg(X_resampled[idx_removed, :], y_resampled[idx_removed])


##################################
### [6] One-sided Selection
##################################

from imblearn.under_sampling import OneSidedSelection
oss = OneSidedSelection(return_indices=True)
X_resampled, y_resampled, idx_resampled = oss.fit_sample(X,y)
idx_removed = np.setdiff1d(np.arange(X_resampled.shape[0]), idx_resampled)
result_dic["One-sided Selection"] = LogisticReg(X_resampled[idx_removed, :], y_resampled[idx_removed])


##################################
### Save result
##################################


methods = ["Non Sampling", "Over Sampling", "Under Sampling", "SMOTE", 
           "Tomek Links", "One-sided Selection"]
result_df = pd.DataFrame()
acc_li, prec_li, rec_li, f1_li = pd.Series(), pd.Series(), pd.Series(), pd.Series()
for m in methods :
    acc_li.set_value(m, result_dic[m][0])
    prec_li.set_value(m, result_dic[m][1])
    rec_li.set_value(m, result_dic[m][2])
    f1_li.set_value(m, result_dic[m][3])
result_df["Accuracy"] = acc_li
result_df["precision"] = prec_li
result_df["recall"] = rec_li
result_df["F1-score"] = f1_li       
    
result_df.to_csv(f_dir+"samplingResult.csv")    
    