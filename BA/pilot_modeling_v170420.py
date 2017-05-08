#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:01:01 2017

@author: seo1ab
"""
import pandas as pd
from pandas import DataFrame as df
import numpy as np
import mysql.connector
import seaborn as sns

################################
### import data table
################################

cnx = mysql.connector.connect(
          user = 'root',
          password = 'master',
          host = '127.0.0.1',
          database = 'project_ba'
          )
#cnx.close()
cursor = cnx.cursor()
query = ("SELECT * FROM offers")
cursor.execute(query)
table_rows = cursor.fetchall()
cursor.close()

df_offers = df(table_rows)
df_offers.columns = [
        'offer', 'category', 'quantity', 'company', 'offervalue', 'brand'
        ]


cursor = cnx.cursor()
query = ("SELECT * FROM trainHistory")
cursor.execute(query)
table_rows = cursor.fetchall()

df_trainHistory = df(table_rows)
df_trainHistory.columns = [
        'id', 'chain', 'offer', 'market', 'repeattrips', 'repeter', 'offeradate'
        ]

"""
cursor = cnx.cursor()
query = ("SELECT * FROM transactions")
cursor.execute(query)
table_rows = cursor.fetchall()

df_transactions = df(table_rows)
df_transactions.columns = [
        'id', 'chain', 'dept', 'category', 'company', 'brand',
        'date', 'productsize', 'productmeasure', 'purchasesequantity',
        'purchaseamount'
        ]


df_transactions = pd.read_csv(
        "/home/seo1ab/workspace/dataset/transactions.csv"
        )
"""
cnx.close()


################################
### data expolation
################################

df_trainHistory.head()
df_offers.head()
df_trainHistory.columns.values
df_offers.columns.values

df_trainHistory['repeattrips'].describe()
df_trainHistory[df_trainHistory.repeter=='t']['repeattrips'].describe()
sns.distplot(df_trainHistory[df_trainHistory.repeter=='t']['repeattrips'])

df_offers['quantity'].describe()
sns.distplot(df_offers['quantity'])
df_offers['quantity'].value_counts()
df_offers['offervalue'].describe()
sns.distplot(df_offers['offervalue'])


################################
### merge table, dummy coding, data split
################################

colnames = ['offer', 'market', 'repeter']
tmp_df = pd.merge(df_trainHistory[colnames], df_offers[['offer', 'offervalue']])
tmp_df = tmp_df.drop('offer', axis=1)
tmp_df.head()

dummy = pd.get_dummies(tmp_df['market'], prefix='market', drop_first=True)
colnames = list(dummy.columns.values)
for col in colnames :
    tmp_df[col] = dummy[col]
tmp_df = tmp_df.drop('market', axis=1)
tmp_df.head()

colnames = list(tmp_df.columns.values)
X = np.array(tmp_df[colnames[1:]])
y = np.array(tmp_df[colnames[0]])

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
train_idx, test_idx = [], []
for train, test in skf.split(X,y) :
    train_idx.append(train)
    test_idx.append(test)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(['t', 'f'])
le.classes_
label_dic = {}
for i in range(len(le.classes_)) :
    label_dic[le.classes_[i]] = i
y = le.transform(y)
label_dic.items()


from sklearn import naive_bayes, metrics
clf = naive_bayes.GaussianNB()
acc_li, prec_li, rec_li, f1_li = [], [], [], []
for i in range(len(test_idx)) :
    clf.fit(X[train_idx[i]], y[train_idx[i]])
    pred = clf.predict(X[test_idx[i]])
    y_ture = y[test_idx[i]]
    acc_li.append(metrics.accuracy_score(y_ture, pred))
    prec_li.append(metrics.precision_score(y_ture, pred))
    rec_li.append(metrics.recall_score(y_ture, pred))
    f1_li.append(metrics.f1_score(y_ture, pred))
    print("do {}/5".format(i+1))
result = { 'Acc' : np.array(acc_li).mean(),
        'precision' : np.array(prec_li).mean(),
        'recall' : np.array(rec_li).mean(),
        'F1' : np.array(f1_li).mean()
        }
print("Acc:\t{}\nprecision:\t{}\nrecall:\t{}\nF1 score:\t{}".format(
        result['Acc'], result['precision'], result['recall'], result['F1']
        ))


