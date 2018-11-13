# 数据分析库
import pandas as pd
# 科学计算库
import numpy as np
from pandas import Series, DataFrame
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

df1 = pd.read_csv("data/train_1.txt", sep='\t')
aa = list(df1.columns.values)[0:6749]
df2 = pd.read_csv("data/train_2.txt", sep='\t', names=aa)
df3 = pd.read_csv("data/train_3.txt", sep='\t', names=aa)
df4 = pd.read_csv("data/train_4.txt", sep='\t', names=aa)
df5 = pd.read_csv("data/train_5.txt", sep='\t', names=aa)
dfp = pd.read_csv("data/valid.txt", sep='\t')
adalist = pd.read_csv("data/feature_top_decisiontree_ExtraTree_list.txt", sep='\t')
#print(adalist)
#data_train.head(5)
frames = [df1, df2, df3, df4, df5]
#frames = [df1]
data_train = pd.concat(frames)
data_train.info()

bb=data_train.iloc[:, 4:6749]
#dd=data_train.iloc[:, 3:4]
cc = bb.apply(lambda x: x.fillna(x.mean()), axis=0)

cc['tag'] = data_train.iloc[:, 3:4]
test = dfp.iloc[:, 2:6747].apply(lambda x: x.fillna(x.mean()), axis=0)
Xtest = adalist
x_data_output = dfp.iloc[:, 0:1].values
#print(data_train.iloc[:, 3:4])
#

predictors = adalist
alg =  RandomForestClassifier(random_state=1, n_estimators=52, min_samples_split=2, min_samples_leaf=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
# kf=cross_validation.KFold(data_train.shape[0],n_folds=10,random_state=1)
kf = model_selection.KFold(n_splits=120, shuffle=False, random_state=1)
scores = model_selection.cross_val_score(alg, cc[predictors], cc['tag'], cv=kf)
print("scores.mean=", scores.mean())

File = open("data/prob_feature_top_decisiontree_ExtraTree.txt", "w",encoding=u'utf-8', errors='ignore')
File.write("id"+"," + "prob" + "\n")
classifier = alg.fit(cc[predictors], cc['tag'])
predictiontest = classifier.predict_proba(test[adalist])[:, 1]
for step in range(len(test[adalist])):
    File.write(str(x_data_output[step][0])+"," + str(predictiontest[step]) + "\n")
#print(predictiontest)

