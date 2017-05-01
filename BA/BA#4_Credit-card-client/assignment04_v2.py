# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:05:25 2017

@author: Hanbin Seo
"""

import numpy as np
import pandas as pd
f_dir = "D:/5eo1ab/"
data =pd.read_csv(f_dir+"UCI_Credit_Card.csv")
copy_data = data.copy()

TARGET_COL = 'default.payment.next.month'

##################################
### Imputation
##################################

colnames = list(data.columns.values)
data.dtypes
for col in colnames :
    print(data[col].isnull().value_counts())
"""
null 인 변수는 없는 것으로 확인됨.
"""

##################################
### Data Explore
##################################

cat_colnames = ['SEX', 'EDUCATION', 'MARRIAGE']
for col in cat_colnames :
    print(">> {0}\n{1}".format(col, data[col].value_counts()))
"""
추가적으로 조작이 필요한 변수 : 'EDUCATION', 'MARRIAGE'
한편 'EDUCATION' 변수에서의 5와 6은 동일한 unknown 변수이므로 통일해야하고, 0은 미확인 값
'MARRIAGE'변수에서는 0은 미확인 값, 의미를 생각하면 3(others)와 같다.
결과적으로  불명확한 변수는 others 변수로 통일해 dummy coding을 진행하고자 함.
"""
pay_colnames = ['PAY_{}'.format(i) for i in range(7) if i!=1]
for col in pay_colnames :
    print(">> {0}\n{1}".format(col, data[col].value_counts()))
"""
일부 변수에서 -1 미만의 값을 갖는 것을 확인할 수 있다. 
또한 0 또한 미확인 값인데, 그 비중이 매우 큰 것을 확인할 수 있다.
하지만 문제에서의 변수 의미를 생각해보면, -2부터 0값은 모두 정상적으로 납부해 -1이라고 가정하는게 좋겠다.
왜냐면 언제 세금을 지연납부했는지에 대한 변수인데, 보통의 월급쟁이는 유리지갑이라 칼같이 걷어가기 때문이다.
"""
data.loc[data.EDUCATION==6, 'EDUCATION'] = 5
replace_dic = {
        'EDUCATION' : [0,5],
        'MARRIAGE' : [0,3]
        }
def set_replace_value(colname, from_v, to_v=0) :
    for v in from_v :
        data.loc[data[colname]==v, colname] = to_v
    return None
for k, v in replace_dic.items() :
    set_replace_value(k, v, to_v=max(v))
for col in pay_colnames :
    set_replace_value(col, [-2,0], to_v=-1)
"""
다음과 같이 변수설명을 수정해야 겠다.
EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown)
"""

##################################
### Dummy Coding
##################################
dm_list = []
dm_colnames = cat_colnames + pay_colnames
for col in dm_colnames :
     dm_list.append(pd.get_dummies(data[col], prefix=col))
data = data.drop(dm_colnames, axis=1)
for dm_col in dm_list :
    data[list(set(dm_col))] = dm_col

##################################
### Normalization
##################################
other_colnames = [c for c in colnames if c not in dm_colnames and c not in ['ID', TARGET_COL]]
for col in other_colnames :
    nor_col = (data[col]-data[col].min()) / (data[col].max()-data[col].min())
    del data[col]
    data[col] = nor_col


##################################
### Input/Output(label) data split
##################################

colnames = [col for col in data.columns.values if col not in ['ID', TARGET_COL]]
X = data[colnames]
y = data[TARGET_COL]




##################################
### Using stratified 5-fold cross-validation for model selection
##################################

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
idx_train, idx_test = [], []
for train, test in skf.split(X, y) :
    idx_train.append(train)
    idx_test.append(test)
    
##################################
### Use support vector machine
##################################

from sklearn import svm
from sklearn import metrics
import time
params = [10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3]
res_dic = {}
i = 1
t0 = time.time()
t1 = t0
for c in params :
    for r in params :
        model = svm.SVC(kernel='rbf', C=c, gamma=r)
        fold_acc, fold_f1 = [], []
        for k in range(len(idx_train)) :
            X_train, y_train = X.iloc[idx_train[k]].values, y.iloc[idx_train[k]].values
            X_test, y_test = X.iloc[idx_test[k]].values, y.iloc[idx_test[k]].values
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = metrics.accuracy_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred)
            fold_acc.append(acc)
            fold_f1.append(f1)
        acc, f1 = np.array(fold_acc).mean(), np.array(fold_f1).mean()
        res_dic[(c,r)] = (acc, f1)
        t1 = time.time() - t1
        print("in process {}/49\t@ {}".format(i, round(t1, 4)))
        i += 1
print("Total processing time: {}".format(round(time.time()-t0, 4)))


##################################
### Save result
##################################

df_acc, df_f1 = pd.DataFrame(), pd.DataFrame()
for c in params :
    Acc_series, F1_series = pd.Series(), pd.Series()
    for r in params :
        Acc_series = Acc_series.set_value(r, res_dic[(c,r)][0])
        F1_series = F1_series.set_value(r, res_dic[(c,r)][1])
    df_acc[c] = Acc_series
    df_f1[c] = F1_series
df_acc.to_csv(f_dir+"result_Acc.csv")
df_f1.to_csv(f_dir+"result_F1.csv")
