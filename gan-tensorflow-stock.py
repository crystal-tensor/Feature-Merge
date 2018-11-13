import tensorflow as tf
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn  import preprocessing
import math
import pymysql
import torch
import torch.nn as nn
tf.set_random_seed(1)
np.random.seed(1)


# Hyper Parameters
BATCH_SIZE = 18521
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 6             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 2     # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

conn = pymysql.Connect(host='192.168.101.74',port=3306,user='root',passwd='123456',db='fin_predict_db',charset='utf8')
cursor = conn.cursor()

sqlcount = "select symbol from fin_sh_price where report_type = 's1'  and  input_data <> 0 and df_revenue <> 0"
sym= pd.read_sql(sql=sqlcount, con=conn)

sqlstock = "select input_data,df_revenue from fin_sh_price where report_type = 's1'  and  input_data <> 0 and df_revenue <> 0 order by symbol"
dfstock = pd.read_sql(sql=sqlstock, con=conn)
x_data = preprocessing.minmax_scale(dfstock, feature_range=(-1, 1))

# PAINT_POINTS = goodx_data
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[:])
# plt.legend(loc='upper right')
# plt.show()

from numpy import random as nr
np.set_printoptions(precision=2)

def predict():
    y = x_data[:, 0:2]
    paintings = torch.from_numpy(y).float()
    return paintings


with tf.variable_scope('Generator'):
    G_in = tf.placeholder(tf.float32, [None, N_IDEAS])          # random ideas (could from normal distribution)
    G_l1 = tf.layers.dense(G_in, 128, tf.nn.relu)
    G_out = tf.layers.dense(G_l1, ART_COMPONENTS)               # making a painting from these random ideas

with tf.variable_scope('Discriminator'):
    real_art = tf.placeholder(tf.float32, [None, ART_COMPONENTS], name='real_in')   # receive art work from the famous artist
    D_l0 = tf.layers.dense(real_art, 128, tf.nn.relu, name='l')
    prob_artist0 = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')              # probability that the art work is made by artist
    # reuse layers for generator
    D_l1 = tf.layers.dense(G_out, 128, tf.nn.relu, name='l', reuse=True)            # receive art work from a newbie like G
    prob_artist1 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True)  # probability that the art work is made by artist

D_loss = -tf.reduce_mean(tf.log(prob_artist0) + tf.log(1-prob_artist1))
G_loss = tf.reduce_mean(tf.log(1-prob_artist1))

train_D = tf.train.AdamOptimizer(LR_D).minimize(D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
train_G = tf.train.AdamOptimizer(LR_G).minimize(G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
File = open("data/D_loss_tensorflow.txt", "w",encoding=u'utf-8', errors='ignore')
#plt.ion()   # something about continuous plotting
for step in range(12000):
    artist_paintings = predict()           # real painting from artist
    G_ideas = np.random.randn(BATCH_SIZE, N_IDEAS)
    G_paintings, pa0, Dl = sess.run([G_out, prob_artist0, D_loss, train_D, train_G],    # train and get results
                                    {G_in: G_ideas, real_art: artist_paintings})[:3]
    #if step % 120 == 0:
        #base_path = saver.save(sess, "module/autobid.model")
    if step % 50 == 0:  # plotting
        File.write(str(sess.run(D_loss, {G_in: G_ideas, real_art: artist_paintings})) + "\n")
        print(sess.run(D_loss, {G_in: G_ideas, real_art: artist_paintings}))
#         plt.cla()
#         plt.plot(PAINT_POINTS[0], G_paintings[0], c='#4AD631', lw=3, label='Generated painting',)
#        # plt.plot(PAINT_POINTS[0], 2 * PAINT_POINTS[0] + np.sin(PAINT_POINTS[0]*10) + 1, c='#74BCFF', lw=3, label='upper bound')
#         plt.plot(PAINT_POINTS[:])
#         plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % pa0.mean(), fontdict={'size': 15})
#         plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -Dl, fontdict={'size': 15})
#         plt.ylim((0, 3)); plt.legend(loc='upper right', fontsize=12); plt.draw(); plt.pause(0.01)
#
# plt.ioff()
# plt.show()


#
# with tf.name_scope('loss'):
#     D_loss = -tf.reduce_mean(tf.log(prob_artist0) + tf.log(1 - prob_artist1))
#     G_loss = tf.reduce_mean(tf.log(1 - prob_artist1))
#     tf.summary.scalar('D_loss', D_loss)
#     tf.summary.scalar('G_loss', G_loss)
#
#
# with tf.name_scope('train'):
#         train_D = tf.train.AdamOptimizer(LR_D).minimize(D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
#         train_G = tf.train.AdamOptimizer(LR_G).minimize(G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))
#
# sess = tf.Session()
# merged = tf.summary.merge_all()
#
# writer = tf.summary.FileWriter("logs/", sess.graph)
#
# init = tf.global_variables_initializer()
# sess.run(init)
#
# for i in range(12000):
#     artist_paintings = earn()  # real painting from artist
#     G_ideas = np.random.randn(BATCH_SIZE, N_IDEAS)
#     sess.run([G_out, prob_artist0, D_loss, train_D, train_G], {G_in: G_ideas, real_art: artist_paintings})
#     if i % 50 == 0:
#         result = sess.run(merged, feed_dict={G_in: G_ideas, real_art: artist_paintings})
#         writer.add_summary(result, i)


# if int((tf.__version__).split('.')[1]) < 12 and int(
#             (tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
#             writer = tf.train.SummaryWriter('logs/', sess.graph)
# else:  # tensorflow version >= 0.12
#             writer = tf.summary.FileWriter("logs/", sess.graph)
#tensorboard --logdir=logs