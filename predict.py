###行情数据对应GB

import pandas as pd
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import math
import keras
import pymysql

# hidden layer
rnn_unit = 128
# feature
input_size = 1
output_size = 1
input_size32 = 32
lr = 0.0006

#conn = pymysql.Connect(host='192.168.101.10',port=3306,user='root',passwd='123456',db='stockcn',charset='utf8')
conn = pymysql.Connect(host='192.168.101.74',port=3306,user='root',passwd='123456',db='fin_predict_db',charset='utf8')
cursor = conn.cursor()
sqlf = "select input_data,symbol from fin_sh_predict  where publish_date > '2018-01-01' group by symbol"
df = pd.read_sql(sql=sqlf, con=conn)
df = df.dropna()
xy_data = preprocessing.minmax_scale(df, feature_range=(-1,1))
#print(df.iloc[2000:2005, 1:2].values)

def addLayer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)   # to control dropout when training and testing

close1 = addLayer(xs, input_size, 1, n_layer=12, activation_function=tf.nn.tanh) # tanh是激励函数的一种
close2 = addLayer(close1, 1, 1, n_layer=12, activation_function=tf.nn.tanh)
#costd2 = tf.layers.dropout(close2, rate=0.1, training=tf_is_training)   # drop out 50% of inputs


#loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - costd2), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
#train = tf.train.GradientDescentOptimizer(lr).minimize(loss)  # 选择梯度下降法


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
File = open("data/predict2018.txt", "w",encoding=u'utf-8', errors='ignore')
with tf.Session() as sess:
    # saver = tf.train.import_meta_graph('module/forcase200092.ckpt.meta')
    # saver.restore(sess, tf.train.latest_checkpoint("module/"))

    saver.restore(sess, "module/forcase2018.ckpt")
    for step in range(len(xy_data)-1):
        prob = sess.run(close2, feed_dict={xs: xy_data[step:step+1, 0:1]})
        File.write(str(df.iloc[step:step+1, 1:2].values))
        File.write(str(prob) + "\n")


