import pandas as pd
import tensorflow as tf
import numpy as np
import math
from pandas import Series, DataFrame
from sklearn import preprocessing

input_size = 6745
output_size = 1
lr = 0.0006

df1 = pd.read_csv("data/train_1.txt", sep='\t')
aa = list(df1.columns.values)[0:6749]
df2 = pd.read_csv("data/train_2.txt", sep='\t', names=aa)
df3 = pd.read_csv("data/train_3.txt", sep='\t', names=aa)
df4 = pd.read_csv("data/train_4.txt", sep='\t', names=aa)
df5 = pd.read_csv("data/train_5.txt", sep='\t', names=aa)
#dfp = pd.read_csv("data/valid.txt", sep='\t')
#adalist = pd.read_csv("data/adaboostfeature2.txt", sep='\t')
#data_train.head(5)
frames = [df1, df2, df3, df4, df5]
#frames = [df1]
data_train = pd.concat(frames)
data_train.info()

# bb = data_train.iloc[:, 4:6749]
# cc = data_train.iloc[:, 3:4]
# #dd=data_train.iloc[:, 3:4]
# dd = bb.apply(lambda x: x.fillna(x.mean()), axis=0)
# ee = cc.apply(lambda x: x.fillna(x.mean()), axis=0)
# #test = dfp.iloc[:, 2:6747].apply(lambda x: x.fillna(x.mean()), axis=0)
# Xtest = adalist
# predictors = adalist


def addLayer(inputData, inSize, outSize, activity_function=None):
    Weights = tf.Variable(tf.random_normal([inSize, outSize]))
    basis = tf.Variable(tf.zeros([1, outSize]) + 0.1)
    weights_plus_b = tf.matmul(inputData, Weights) + basis
    if activity_function is None:
        ans = weights_plus_b
    else:
        ans = activity_function(weights_plus_b)
    return ans

xxx = data_train.iloc[:, 4:6749].apply(lambda x: x.fillna(x.mean()), axis=0)
yyy = data_train.iloc[:, 3:4].apply(lambda x: x.fillna(x.mean()), axis=0)

x_data = preprocessing.minmax_scale(xxx.iloc[:, 0:].values, feature_range=(-1,1))
y_data = preprocessing.minmax_scale(yyy.iloc[:, 0:].values, feature_range=(-1,1))


xs = tf.placeholder(tf.float32, [None, input_size])  # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
ys = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing


l1 = addLayer(xs, input_size, 1, activity_function=tf.nn.relu) # relu是激励函数的一种
d1 = tf.layers.dropout(l1, rate=0.1, training=tf_is_training)
l2 = addLayer(l1, 1, 1, activity_function=None)
d2 = tf.layers.dropout(l2, rate=0.1, training=tf_is_training)   # drop out 50% of inputs

loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys - l2)), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
train = tf.train.GradientDescentOptimizer(lr).minimize(loss)  # 选择梯度下降法

d_out = tf.layers.dense(d2, 1)
d_loss = tf.losses.mean_squared_error(ys, d_out)
d_train = tf.train.AdamOptimizer(lr).minimize(d_loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

for i in range(12000):
    #d_loss, d_train = sess.run([d_loss, d_train], feed_dict={xs: x_data, ys: y_data,tf_is_training: True})
    loss_overfiting, trainr = sess.run([loss, train], feed_dict={xs: x_data, ys: y_data,tf_is_training: True})
    if i % 12000 == 0:
        base_path = saver.save(sess, "module/360_tf.model")
    if i % 600 == 0:
        print("loss",sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        #print("loss_dropout", sess.run(d_loss, feed_dict={xs: x_data, ys: y_data,tf_is_training: True}))

# File = open("data/prob_adaboostfeature.txt", "w",encoding=u'utf-8', errors='ignore')
# File.write("id"+",")
# File.write("prob" + "\n")
# classifier = alg.fit(cc[predictors], cc['tag'])
# predictiontest = classifier.predict_proba(test[Xtest])
# for step in range(len(test[Xtest])):
#     File.write(str(x_data_output[step])+",")
#     File.write(str(predictiontest[step]) + "\n")
#print(predictiontest)


#print(scores)
# Take the mean of the scores (because we have one for each fold)


