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
sqlf = "select input_data,df_revenue,symbol from fin_sh_price where report_type = 's1'  and  input_data <> 0 and df_revenue <> 0 and df_revenue <10 order by symbol"
df = pd.read_sql(sql=sqlf, con=conn)
df = df.dropna()
xy_data = preprocessing.minmax_scale(df, feature_range=(-1,1))
#     stock = preprocessing.minmax_scale(stockdata,feature_range=(-1,1))
#     print(stock)

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

def financeforcase():
    close1 = addLayer(xs, input_size, 1, n_layer=12, activation_function=tf.nn.tanh) # tanh是激励函数的一种
    close2 = addLayer(close1, 1, 1, n_layer=12, activation_function=tf.nn.tanh)
    costd2 = tf.layers.dropout(close2, rate=0.1, training=tf_is_training)   # drop out 50% of inputs
    return close2

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - financeforcase()), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
train = tf.train.GradientDescentOptimizer(lr).minimize(loss)  # 选择梯度下降法

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
#File = open("data/stockforcase.txt", "w+",encoding=u'utf-8', errors='ignore')

for i in range(20000):
    loss_financeforcase, train_emotion = sess.run([loss, train], feed_dict={xs: xy_data[:, 0:1], ys: xy_data[:, 1:2], tf_is_training: True})
    # if i % 50 == 0:
    #     File.write(str(loss.data) + "\n")
    if i % 20000 == 0:
        print("loss_aplha", sess.run(loss, feed_dict={xs: xy_data[:, 0:1], ys: xy_data[:, 1:2], tf_is_training: True}))
        base_path = saver.save(sess, "module/forcase2018.ckpt")
    # if i % 20000 == 0:
    #     #print("loss_aplha", sess.run(loss, feed_dict={xs: xy_data[:, 0:1], ys: xy_data[:, 1:2],  tf_is_training: True}))
    #     base_path = saver.save(sess, "module/forcase2018.model")




with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - financeforcase()), reduction_indices=[1]))
    tf.summary.scalar('finance-forcase', loss)


with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

sess = tf.Session()
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(20000):
    sess.run(train_step, feed_dict={xs: xy_data[:, 0:1], ys: xy_data[:, 1:2], tf_is_training: True})
    if i % 50 == 0:
        result = sess.run(merged,feed_dict={xs: xy_data[:, 0:1], ys: xy_data[:, 1:2], tf_is_training: True})
        writer.add_summary(result, i)


# if int((tf.__version__).split('.')[1]) < 12 and int(
#             (tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
#             writer = tf.train.SummaryWriter('logs/', sess.graph)
# else:  # tensorflow version >= 0.12
#             writer = tf.summary.FileWriter("logs/", sess.graph)
#tensorboard --logdir=logs