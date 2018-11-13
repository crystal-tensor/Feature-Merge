import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
from sklearn import model_selection


#data_test = pd.read_csv("F:\\金融算法挑战\\360金融算法挑战赛\\open_data_train_valid\\valid.txt", sep='\t')
df1 = pd.read_csv("data/train_1.txt", sep='\t')
aa = list(df1.columns.values)[0:6749]
df2 = pd.read_csv("data/train_2.txt", sep='\t', names=aa)
df3 = pd.read_csv("data/train_3.txt", sep='\t', names=aa)
df4 = pd.read_csv("data/train_4.txt", sep='\t', names=aa)
df5 = pd.read_csv("data/train_5.txt", sep='\t', names=aa)

#data_train.head(5)
frames = [df1, df2, df3, df4, df5]
#frames = [df1]
data_train = pd.concat(frames)
data_train.info()

bb=data_train.iloc[:, 4:6749]
#dd=data_train.iloc[:, 3:4]
xx = bb.apply(lambda x: x.fillna(x.mean()), axis=0)
yy = data_train.iloc[:, 3:4]
# #

clf1 = KNeighborsClassifier(3)
clf2 = SVC(probability=True)
clf3 = DecisionTreeClassifier()
clf4 = RandomForestClassifier()
clf5 = AdaBoostClassifier()
clf6 = GradientBoostingClassifier()
clf7 = GaussianNB()
clf8 = LinearDiscriminantAnalysis()
clf9 = QuadraticDiscriminantAnalysis()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf4,clf5, clf6, clf7,clf8, clf9],
                          meta_classifier=lr)


xx = data_train.iloc[:, 4:6749].apply(lambda x: x.fillna(x.mean()), axis=0)
yy = data_train.iloc[:, 3:4]
trainx = xx.values
trainy = yy.values
X = trainx[0::, 1::]
y = trainy[0::, 0]

for clf, label in zip([clf1, clf2, clf4,clf5, clf6, clf7,clf8, clf9, sclf],
                      ['KNeighborsClassifier',
                       'SVC',
                       'DecisionTreeClassifier',
                       'RandomForestClassifier',
                       'AdaBoostClassifier',
                       'GradientBoostingClassifier',
                       'GaussianNB',
                       'LinearDiscriminantAnalysis',
                       'QuadraticDiscriminantAnalysis',]):

    scores = model_selection.cross_val_score(clf, X, y,
                                             cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))


