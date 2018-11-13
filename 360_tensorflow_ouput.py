import pandas as pd
import tensorflow as tf
import numpy as np
import math
from pandas import Series, DataFrame
from sklearn import preprocessing

input_size = 6745
output_size = 1
lr = 0.0006

df = pd.read_csv("data/valid.txt", sep='\t')
xxx = df.iloc[:, 2:6747].apply(lambda x: x.fillna(x.mean()), axis=0)
x_data = preprocessing.minmax_scale(xxx.iloc[:, 0:].values, feature_range=(-1,1))
x_data_output = df.iloc[:, 0:1].values

xs = tf.placeholder(tf.float32, [None, input_size])  # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
ys = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing

def addLayer(inputData, inSize, outSize, activity_function=None):
    Weights = tf.Variable(tf.random_normal([inSize, outSize]))
    basis = tf.Variable(tf.zeros([1, outSize]) + 0.1)
    weights_plus_b = tf.matmul(inputData, Weights) + basis
    if activity_function is None:
        ans = weights_plus_b
    else:
        ans = activity_function(weights_plus_b)
    return ans


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

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
File = open("data/360-tensorflow-output.txt", "w",encoding=u'utf-8', errors='ignore')
with tf.Session() as sess:
    saver.restore(sess, "module/360_tf.model")
    for step in range(len(x_data)):
        prob = sess.run(l2, feed_dict={xs: [x_data[step]]})
        File.write(str(x_data_output[step]))
        File.write(str(prob) + "\n")
print("it's done")


