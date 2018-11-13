import pandas as pd
import gc
import numpy as np
from pandas import Series, DataFrame
from sklearn  import preprocessing
from sklearn import model_selection
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import ColumnSelector
from sklearn.ensemble import GradientBoostingClassifier

df1 = pd.read_csv("data/train_1.txt", sep='\t')
aa = list(df1.columns.values)[0:6749]
df2 = pd.read_csv("data/train_2.txt", sep='\t', names=aa)
df3 = pd.read_csv("data/train_3.txt", sep='\t', names=aa)
df4 = pd.read_csv("data/train_4.txt", sep='\t', names=aa)
df5 = pd.read_csv("data/train_5.txt", sep='\t', names=aa)
dfp = pd.read_csv("data/valid.txt", sep='\t')
#print(adalist)
#data_train.head(5)
frames = [df1, df2, df3, df4, df5]
#frames = [df1]
data_train = pd.concat(frames)
data_train.info()

del df5
del df4
del df3
del df2
del df1
gc.collect()

bb=data_train.iloc[:, 4:6749]
#dd=data_train.iloc[:, 3:4]
cc = bb.apply(lambda x: x.fillna(x.mean()), axis=0)
x_data = preprocessing.minmax_scale(cc.iloc[:, :].values, feature_range=(-1,1))
cc['tag'] = data_train.iloc[:, 3:4]
test = dfp.iloc[:, 2:6747].apply(lambda x: x.fillna(x.mean()), axis=0)
#test_data = preprocessing.minmax_scale(test.iloc[:, :].values, feature_range=(-1,1))
Xtest = list(test.columns.values)[4:6745]
x_data_output = dfp.iloc[:, 0:1].values
#print(data_train.iloc[:, 3:4])
#

predictors = list(cc.columns.values)[4:6745]
alg1 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=42, max_depth=-1, learning_rate=0.054, n_estimators=490,
                               subsample_for_bin=200000, objective='binary', class_weight=None, min_split_gain=0.0,
                               min_child_weight=1, min_child_samples=21, subsample=0.72, subsample_freq=1,
                               colsample_bytree=0.63, reg_alpha=6.18, reg_lambda=2.718, random_state=142857, n_jobs=-1,
                               silent=True, importance_type='split')
alg2 = XGBClassifier(n_estimators=60,max_depth=9,min_child_weight=2,gamma=0.9,subsample=0.8,learning_rate=0.02,
                    colsample_bytree=0.8,objective='binary:logistic',nthread=-1,scale_pos_weight=1)
alg3 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=600, max_depth=7, min_samples_leaf=60,
                           min_samples_split=1200, max_features=9, subsample=0.7, random_state=10)
lr = LogisticRegression()

pipe1 = make_pipeline(ColumnSelector(cols=predictors[4:6745]), lr)
pipe2 = make_pipeline(ColumnSelector(cols=predictors[4:6745]), alg2)
pipe3 = make_pipeline(ColumnSelector(cols=predictors[4:6745]), alg3)

sclf = StackingClassifier(classifiers=[lr, pipe2, pipe3], meta_classifier=alg1)

# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
# kf=cross_validation.KFold(data_train.shape[0],n_folds=10,random_state=1)
kf = model_selection.KFold(n_splits=120, shuffle=False, random_state=1)
scores = model_selection.cross_val_score(sclf, cc[predictors], cc['tag'], cv=kf)
print("scores.mean=", scores.mean())

File = open("data/prob_stackingLXG.txt", "w", encoding=u'utf-8', errors='ignore')
File.write("id"+"," + "prob" + "\n")
classifier = sclf.fit(cc[predictors], cc['tag'])
predictiontest = classifier.predict_proba(test[Xtest])[:, 1]
for step in range(len(test)):
    File.write(str(x_data_output[step][0])+"," + str(predictiontest[step]) + "\n")

#print(predictiontest)

#scores.mean = 0.7515322319156265  n_splits=12
#scores.mean= 0.8014660420523126  num_leaves=31, max_depth=9, learning_rate=0.001, n_estimators=100, predictors[0:6745]
""""
predictors = list(cc.columns.values)[0:6745]
alg1 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=9, learning_rate=0.001, n_estimators=100,
                               subsample_for_bin=200000, objective='binary', metric='binary_logloss',
                               class_weight=None, min_split_gain=0.0,
                               min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=1,
                               colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1,
                               silent=True, importance_type='split')
alg2 = XGBClassifier(n_estimators=60,max_depth=9,min_child_weight=2,gamma=0.9,subsample=0.8,learning_rate=0.02,
                    colsample_bytree=0.8,objective='binary:logistic',nthread=-1,scale_pos_weight=1)
alg3 = RandomForestClassifier(random_state=1, n_estimators=62, min_samples_split=2, min_samples_leaf=1)
lr = LogisticRegression()

pipe1 = make_pipeline(ColumnSelector(cols=predictors[0:6745]), alg1)
pipe2 = make_pipeline(ColumnSelector(cols=predictors[0:6745]), alg2)
pipe3 = make_pipeline(ColumnSelector(cols=predictors[0:6745]), alg3)

sclf = StackingClassifier(classifiers=[pipe1, pipe2, pipe3],
                          meta_classifier=lr)
"""
#scores.mean= 0.6955935251798561
"""""
predictors = list(cc.columns.values)[4:6745]
alg1 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=90, max_depth=-1, learning_rate=0.001, n_estimators=100,
                               subsample_for_bin=200000, objective='binary', metric='binary_logloss',
                               class_weight=None, min_split_gain=0.0,
                               min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=1,
                               colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1,
                               silent=True, importance_type='split')
alg2 = XGBClassifier(n_estimators=60,max_depth=9,min_child_weight=2,gamma=0.9,subsample=0.8,learning_rate=0.02,
                    colsample_bytree=0.8,objective='binary:logistic',nthread=-1,scale_pos_weight=1)
alg3 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=600, max_depth=7, min_samples_leaf=60,
                           min_samples_split=1200, max_features=9, subsample=0.7, random_state=10)
lr = LogisticRegression()

pipe1 = make_pipeline(ColumnSelector(cols=predictors[4:6745]), lr)
pipe2 = make_pipeline(ColumnSelector(cols=predictors[4:6745]), alg2)
pipe3 = make_pipeline(ColumnSelector(cols=predictors[4:6745]), alg3)

sclf = StackingClassifier(classifiers=[lr, pipe2, pipe3], meta_classifier=alg1)
"""
#scores.mean= 0.833844425060192
"""""
predictors = list(cc.columns.values)[4:6745]
alg1 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=90, max_depth=-1, learning_rate=0.1, n_estimators=100,
                               subsample_for_bin=200000, objective='binary', metric='binary_logloss',
                               class_weight=None, min_split_gain=0.0,
                               min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=1,
                               colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1,
                               silent=True, importance_type='split')
alg2 = XGBClassifier(n_estimators=60,max_depth=9,min_child_weight=2,gamma=0.9,subsample=0.8,learning_rate=0.02,
                    colsample_bytree=0.8,objective='binary:logistic',nthread=-1,scale_pos_weight=1)
alg3 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=600, max_depth=7, min_samples_leaf=60,
                           min_samples_split=1200, max_features=9, subsample=0.7, random_state=10)
lr = LogisticRegression()

pipe1 = make_pipeline(ColumnSelector(cols=predictors[4:6745]), lr)
pipe2 = make_pipeline(ColumnSelector(cols=predictors[4:6745]), alg2)
pipe3 = make_pipeline(ColumnSelector(cols=predictors[4:6745]), alg3)

sclf = StackingClassifier(classifiers=[lr, pipe2, pipe3], meta_classifier=alg1)
"""