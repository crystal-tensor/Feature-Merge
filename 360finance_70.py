# 数据分析库
import pandas as pd
# 科学计算库
import numpy as np
from pandas import Series, DataFrame
#data_test = pd.read_csv("F:\\金融算法挑战\\360金融算法挑战赛\\open_data_train_valid\\valid.txt", sep='\t')
df1 = pd.read_csv("data/train_1.txt", sep='\t')
aa = list(df1.columns.values)[0:6749]
df2 = pd.read_csv("data/train_2.txt", sep='\t', names=aa)
df3 = pd.read_csv("data/train_3.txt", sep='\t', names=aa)
df4 = pd.read_csv("data/train_4.txt", sep='\t', names=aa)
df5 = pd.read_csv("data/train_5.txt", sep='\t', names=aa)

#data_train.head(5)
frames = [df1, df2, df3, df4, df5]
data_train = pd.concat(frames)
data_train.info()

bb=data_train.iloc[:, 4:6749]
#dd=data_train.iloc[:, 3:4]
cc = bb.apply(lambda x: x.fillna(x.mean()), axis=0)
cc['tag']= data_train.iloc[:, 3:4]
# #
#print(data_train.iloc[:, 3:4])
#
#
# # 线性回归
from sklearn.linear_model import LinearRegression

# 训练集交叉验证，得到平均值
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold

# 选取简单的可用输入特征
predictors = list(cc.columns.values)[4:6749]

# 初始化现行回归算法
alg = LinearRegression()
# 样本平均分成3份，3折交叉验证
# kf = KFold(cc.shape[0],n_folds=3,random_state=1)
kf = KFold(n_splits=3, shuffle=False, random_state=1)

predictions = []
for train, test in kf.split(cc):
    # The predictors we're using to train the algorithm.  Note how we only take then rows in the train folds.
    train_predictors = (cc[predictors].iloc[train, :])
    # The target we're using to train the algorithm.
    train_target = cc["tag"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(cc[predictors].iloc[test, :])
    predictions.append(test_predictions)



# The predictions are in three aeparate numpy arrays.	Concatenate them into one.
# We concatenate them on axis 0,as they only have one axis.
predictions = np.concatenate(predictions, axis=0)
# Map predictions to outcomes(only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
accuracy = sum(predictions[predictions == cc["tag"]]) / len(predictions)
print("准确率为: ", accuracy)
#
# from sklearn import model_selection
# #逻辑回归
# from sklearn.linear_model import LogisticRegression
# #初始化逻辑回归算法
# alg=LogisticRegression(random_state=1)
# #使用sklearn库里面的交叉验证函数获取预测准确率分数
# scores = model_selection.cross_val_score(alg,data_train[predictors],data_train["tag"],cv=3)
# #使用交叉验证分数的平均值作为最终的准确率
# print("准确率为: ",scores.mean())