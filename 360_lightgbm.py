# 数据分析库
import pandas as pd
# 科学计算库
import numpy as np
from pandas import Series, DataFrame
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

#data_test = pd.read_csv("F:\\金融算法挑战\\360金融算法挑战赛\\open_data_train_valid\\valid.txt", sep='\t')
df1 = pd.read_csv("data/train_1.txt", sep='\t')
aa = list(df1.columns.values)[0:6749]
df2 = pd.read_csv("data/train_2.txt", sep='\t', names=aa)
df3 = pd.read_csv("data/train_3.txt", sep='\t', names=aa)
df4 = pd.read_csv("data/train_4.txt", sep='\t', names=aa)
df5 = pd.read_csv("data/train_5.txt", sep='\t', names=aa)
dfp = pd.read_csv("data/valid.txt", sep='\t')
#adalist = pd.read_csv("data/adaboostfeature2.txt", sep='\t')
#data_train.head(5)
frames = [df1, df2, df3, df4, df5]
#frames = [df1]
data_train = pd.concat(frames)
data_train.info()
#data_train = data_train2.dropna()
bb=data_train.iloc[:, 4:6749]
#dd=data_train.iloc[:, 3:4]
cc = bb.apply(lambda x: x.fillna(x.mean()), axis=0)
cc['tag'] = data_train.iloc[:, 3:4]
test = dfp.iloc[:, 2:6747].apply(lambda x: x.fillna(x.mean()), axis=0)
Xtest = list(test.columns.values)[4:6745]
x_data_output = dfp.iloc[:, 0:1].values
#print(data_train.iloc[:, 3:4])
#
#

predictors = list(cc.columns.values)[4:6745]
# 62棵决策树，停止的条件：样本个数为2，叶子节点个数为1
alg = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=42, max_depth=-1, learning_rate=0.054, n_estimators=490,
                               subsample_for_bin=200000, objective='binary', class_weight=None, min_split_gain=0.0,
                               min_child_weight=1, min_child_samples=21, subsample=0.72, subsample_freq=1,
                               colsample_bytree=0.63, reg_alpha=6.18, reg_lambda=2.718, random_state=142857, n_jobs=-1,
                               silent=True, importance_type='split')
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
# kf=cross_validation.KFold(data_train.shape[0],n_folds=10,random_state=1)
kf = model_selection.KFold(n_splits=120, shuffle=False, random_state=1)
scores = model_selection.cross_val_score(alg, cc[predictors], cc['tag'], cv=kf)
print("scores.mean=", scores.mean())

File = open("data/prob_lightgbm618.txt", "w",encoding=u'utf-8', errors='ignore')
File.write("id"+"," + "prob" + "\n")
classifier = alg.fit(cc[predictors], cc['tag'])
predictiontest = classifier.predict_proba(test[Xtest])[:, 1]
for step in range(len(test[Xtest])):
    File.write(str(x_data_output[step][0])+"," + str(predictiontest[step]) + "\n")
#print(predictiontest)


#print(scores)
# Take the mean of the scores (because we have one for each fold)
#scores.mean= 0.8600615291107446 n_splits=60 num_leaves=33  subsample_for_bin=200000
#scores.mean= 0.8649420679542801 n_splits=120 num_leaves=90
#scores.mean= 0.8270048067188507  n_splits=120 list(cc.columns.values)[5200:6745] num_leaves=90, max_depth=-1, learning_rate=0.1
# scores.mean= 0.6955935251798561 n_splits=120 num_leaves=90, max_depth=9, learning_rate=0.001, n_estimators=100
#scores.mean= 0.6955935251798561  n_splits=120 num_leaves=512, max_depth=9, learning_rate=0.001, n_estimators=100,
#scores.mean= 0.6955935251798561 n_splits=120 num_leaves=90 max_depth=-1, learning_rate=0.001, n_estimators=100
#scores.mean= 0.864612223786397 n_splits=120 num_leaves=90, max_depth=-1, learning_rate=0.1, n_estimators=100 [4:6745]
#scores.mean= 0.8681200897625238
""""
alg = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=36, max_depth=-1, learning_rate=0.1, n_estimators=810,
                              subsample_for_bin=200000, objective='binary', class_weight=None, min_split_gain=0.0,
                               min_child_weight=1, min_child_samples=21, subsample=0.72, subsample_freq=1,
                               colsample_bytree=0.63, reg_alpha=0.0, reg_lambda=0.0, random_state=142857, n_jobs=-1,
                               silent=True, importance_type='split')
"""
#scores.mean= 0.8688011895405644  0.6104
""""
alg = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=42, max_depth=-1, learning_rate=0.054, n_estimators=490,
                               subsample_for_bin=200000, objective='binary', class_weight=None, min_split_gain=0.0,
                               min_child_weight=1, min_child_samples=21, subsample=0.72, subsample_freq=1,
                               colsample_bytree=0.63, reg_alpha=6.18, reg_lambda=2.718, random_state=142857, n_jobs=-1,
                               silent=True, importance_type='split')
"""
#scores.mean= 0.8691508138027395   0.6065
"""
alg = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=42, max_depth=-1, learning_rate=0.054, n_estimators=490,
                               subsample_for_bin=200000, objective='binary', class_weight=None, min_split_gain=0.0,
                               min_child_weight=1, min_child_samples=21, subsample=0.72, subsample_freq=1,
                               colsample_bytree=0.63, reg_alpha=0.618, reg_lambda=0.2718, random_state=142857, n_jobs=-1,
                               silent=True, importance_type='split')
"""
#scores.mean= 0.868180725527621  0.6113
"""
alg = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=36, max_depth=-1, learning_rate=0.1, n_estimators=490,
                               subsample_for_bin=200000, objective='binary', class_weight=None, min_split_gain=0.0,
                               min_child_weight=1, min_child_samples=21, subsample=0.72, subsample_freq=1,
                               colsample_bytree=0.63, reg_alpha=0.02718, reg_lambda=0.0618, random_state=142857, n_jobs=-1,
                               silent=True, importance_type='split')
"""