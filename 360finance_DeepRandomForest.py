# 数据分析库
import pandas as pd
# 科学计算库
import numpy as np
from pandas import Series, DataFrame
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

#data_test = pd.read_csv("F:\\金融算法挑战\\360金融算法挑战赛\\open_data_train_valid\\valid.txt", sep='\t')
df1 = pd.read_csv("data/train_1.txt", sep='\t')
# aa = list(df1.columns.values)[0:6749]
# df2 = pd.read_csv("data/train_2.txt", sep='\t', names=aa)
# df3 = pd.read_csv("data/train_3.txt", sep='\t', names=aa)
# df4 = pd.read_csv("data/train_4.txt", sep='\t', names=aa)
# df5 = pd.read_csv("data/train_5.txt", sep='\t', names=aa)
dfp = pd.read_csv("data/valid.txt", sep='\t')
#data_train.head(5)
#frames = [df1, df2, df3, df4, df5]
frames = [df1]
data_train = pd.concat(frames)
data_train.info()

bb=data_train.iloc[:, 4:6749]
#dd=data_train.iloc[:, 3:4]
cc = bb.apply(lambda x: x.fillna(x.mean()), axis=0)
cc['tag'] = data_train.iloc[:, 3:4]
test = dfp.iloc[:, 2:6747].apply(lambda x: x.fillna(x.mean()), axis=0)
Xtest = list(test.columns.values)[0:6745]
x_data_output = dfp.iloc[:, 0:1].values
#print(data_train.iloc[:, 3:4])
#
#

predictors = list(cc.columns.values)[0:6745]
# 62棵决策树，停止的条件：样本个数为2，叶子节点个数为1
alg = RandomForestClassifier(random_state=1, n_estimators=62, min_samples_split=2, min_samples_leaf=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
# kf=cross_validation.KFold(data_train.shape[0],n_folds=10,random_state=1)
kf = model_selection.KFold(n_splits=12, shuffle=False, random_state=1)
scores = model_selection.cross_val_score(alg, cc[predictors], cc['tag'], cv=kf)
print("scores.mean=", scores.mean())

File = open("data/probsucess.txt", "w",encoding=u'utf-8', errors='ignore')
File.write("ID"+",")
File.write("prob" + "\n")
classifier = alg.fit(cc[predictors], cc['tag'])
predictiontest = classifier.predict_proba(test)
for step in range(len(test)):
    File.write(str(x_data_output[step])+",")
    File.write(str(predictiontest[step]) + "\n")
#print(predictiontest)


#print(scores)
# Take the mean of the scores (because we have one for each fold)


#scores.mean= 0.9040360798497216



