import pandas as pd
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
adalist = pd.read_csv("data/adaboostfeature2.txt", sep='\t')

frames = [df1, df2, df3, df4, df5]

data_train = pd.concat(frames)
data_train.info()

bb=data_train.iloc[:, 4:6749]

cc = bb.apply(lambda x: x.fillna(x.mean()), axis=0)
cc['tag'] = data_train.iloc[:, 3:4]
test = dfp.iloc[:, 2:6744].apply(lambda x: x.fillna(x.mean()), axis=0)
Xtest = adalist
x_data_output = dfp.iloc[:, 0:1].values

predictors = adalist


alg = RandomForestClassifier(random_state=1, n_estimators=62, min_samples_split=2, min_samples_leaf=1)

kf = model_selection.KFold(n_splits=33, shuffle=False, random_state=1)
scores = model_selection.cross_val_score(alg, cc[predictors], cc['tag'], cv=kf)
print("scores.mean=", scores.mean())

File = open("data/prob_radomforest_features.txt", "w",encoding=u'utf-8', errors='ignore')
File.write("id"+",")
File.write("prob" + "\n")
classifier = alg.fit(cc[predictors], cc['tag'])
predictiontest = classifier.predict_proba(test)
for step in range(len(test)):
    File.write(str(x_data_output[step])+",")
    File.write(str(predictiontest[step]) + "\n")


