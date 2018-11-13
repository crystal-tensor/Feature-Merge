#预测收益：现在的价格-市场的波动=未来的收益

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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

conn = pymysql.Connect(host='192.168.101.10',port=3306,user='root',passwd='123456',db='stockcn_backup',charset='utf8')
cursor = conn.cursor()
#现金流量表
sql = "select * from `cash flow statement_general business`  where `TICKER_SYMBOL_股票代码` = 600222"
df = pd.read_sql(sql=sql, con=conn)
cashflow = preprocessing.minmax_scale(df, feature_range=(-1, 1))
print(df)
emotion = addLayer(cashflow, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)
#balance
sql = "select * from `balance sheet_general business`  where `TICKER_SYMBOL_股票代码` = 600222"
df = pd.read_sql(sql=sql, con=conn)
balance = preprocessing.minmax_scale(df, feature_range=(-1, 1))
print(df)
emotion = addLayer(balance, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)
#income
sql = "select * from `income statement_general business`  where `TICKER_SYMBOL_股票代码` = 600222"
df = pd.read_sql(sql=sql, con=conn)
income = preprocessing.minmax_scale(df, feature_range=(-1, 1))
print(df)
emotion = addLayer(income, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)








#
#
# #环境评估
# def environment():     # painting from the famous artist (real target)
#     rein_conscions = outputLayer(emotion / reward, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)  # 结果和初心的距离
#     # 行为强化 action = outputLayer(reward / moveloss, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)   # 行为强化
#     output_logic = outputLayer(firstmind / ar, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)  # 结果和推理的距离
#     result_alpha = outputLayer(ar, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)  # 分类和推理的距离  任何的分类都是一种方法
#     position = addLayer(output_logic / result_alpha, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)
#     return position
# # correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# # accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# loss_position = tf.reduce_mean(tf.reduce_sum(tf.square((ys - environment())), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
# train_position = tf.train.GradientDescentOptimizer(lr).minimize(loss_position)
#
# #自我评估：
# def selfassess():     # painting from the famous artist (real target)
#     new_cloass = addLayer(stock / firstmind, 1, 1, n_layer=1, activation_function=tf.nn.tanh)
#     newconscuous = addLayer(firstmind * moveloss * new_cloass, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)
#     #意识强化
#     reinforceconscious = addLayer(firstmind / (reward - emotion), input_size, 1, n_layer=1,activation_function=tf.nn.tanh)
#     unconscious = addLayer(firstmind * reinforceconscious * newconscuous, input_size, 1, n_layer=1,activation_function=tf.nn.tanh)
#     wisdow = addLayer(unconscious / reinforceconscious, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)
#     return wisdow