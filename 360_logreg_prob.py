# 数据分析库
import pandas as pd
# 科学计算库
import numpy as np
from pandas import Series, DataFrame
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#data_test = pd.read_csv("F:\\金融算法挑战\\360金融算法挑战赛\\open_data_train_valid\\valid.txt", sep='\t')
df1 = pd.read_csv("data/train_1.txt", sep='\t')
aa = list(df1.columns.values)[0:6749]
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
Xtest = list(test.columns.values)[2:6747]
# #
#print(data_train.iloc[:, 3:4])
#
#


predictors = []
alg=LogisticRegression(random_state=1)
#使用sklearn库里面的交叉验证函数获取预测准确率分数
scores = model_selection.cross_val_score(alg, cc[predictors], cc['tag'], cv=3)
print(alg)
#使用交叉验证分数的平均值作为最终的准确率
print("准确率为: ", scores.mean())

logreg = LogisticRegression()
logreg.fit(cc[predictors], cc['tag'])
Y_pred = logreg.predict(test)
print(Y_pred)


