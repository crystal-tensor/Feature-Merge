import pandas as pd
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from sklearn  import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import math
import keras
import pymysql
import torch
import torch.nn as nn

BATCH_SIZE = 1000
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 6             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 2     # it could be total point G can draw in the canvas


conn = pymysql.Connect(host='192.168.101.74',port=3306,user='root',passwd='123456',db='fin_predict_db',charset='utf8')
cursor = conn.cursor()

sqlstock = "select input_data,df_revenue,symbol from fin_sh_price where report_type = 's1'  and  input_data <> 0 and df_revenue <> 0 order by symbol limit 1000"
dfstock = pd.read_sql(sql=sqlstock, con=conn)
x_data = preprocessing.minmax_scale(dfstock, feature_range=(-1, 1))

sqlcount = "select symbol from fin_sh_price where report_type = 's1'  and  input_data <> 0 and df_revenue <> 0 order by symbol limit 1000"
sym= pd.read_sql(sql=sqlcount, con=conn)
#print(dfxdate[0:10][0:10])

from numpy import random as nr
np.set_printoptions(precision=2)

def predict():
    y = x_data[:, 0:2]
    paintings = torch.from_numpy(y).float()
    return paintings


G = nn.Sequential(                      # Generator
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),
)

D = nn.Sequential(                      # Discriminator
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
#File = open("data/d_loss.txt", "w",encoding=u'utf-8', errors='ignore')
File = open("data/D_loss_stock.txt", "w+",encoding=u'utf-8', errors='ignore')

for step in range(10000):
    artist_paintings = predict()           # real painting from artist
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)  # random ideas
    G_paintings = G(G_ideas)                    # fake painting from G (random ideas)
   # print(G_ideas)
    prob_artist0 = D(artist_paintings)          # D try to increase this prob
    prob_artist1 = D(G_paintings)               # D try to reduce this prob

    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    G_loss = -torch.mean(torch.log(1. - prob_artist1))

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)      # reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
    if step % 50 == 0:
    #     #print('D_loss', "%.6f"%D_loss.data)
    #     # print('G_loss', G_loss)
    #     # print("\n")
         File.write(str("%.6f"%D_loss.data)+" , "+str("%.6f"%G_loss.data) + "\n")
    if step % 10000 == 0:
        torch.save(G, 'module/stock.pkl')
         # base_path = saver.save(sess, "module/bid_forcase.model")
    #     #print(G_paintings)  dfxdate
    #       for i in range(BATCH_SIZE):
    #            File.write(str(sym[i:i+1][0:1].values))
    #            File.write(str("%.6f"%G_paintings[i][0].data)+" , "+str("%.6f"%G_paintings[i][1].data)+"\n")


#tensorboard --logdir=logs