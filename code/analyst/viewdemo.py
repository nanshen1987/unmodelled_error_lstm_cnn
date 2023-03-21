import os

import numpy as np

from analyst.viewhandler import  viewnedpreproc, viewnedpreprocoverview, viewallpreproc

from config.configutil import getpath

# viewnedpreprocoverview(getpath('preproced_path_each'))

# for root, _, wfiles in os.walk(getpath('preprocpath','alic')):
#     for file in wfiles:
#         viewnedpreproc(getpath('preprocpath','alic'), file)
# viewcnnmultimodel('3_day_cnnmulti')
# 1_1
# preproced_path = getpath('preproced_path')
# trained_path = getpath("trained_path")
# filenames = []
# wfiles = os.listdir(preproced_path)
# for file in wfiles:
#         filenames.append(os.path.join(preproced_path, file))
# filenames.reverse()
# data = np.load(filenames[0])


# 1_2 view cnn
# viewsolemodel('north' + "45_day_cnn", 1)
# viewsolemodel('east' + "44_day_cnn", 2)
# viewsolemodel('up' + "14_day_cnn", 3)
# viewcnnmultimodel('3_day_cnnmulti')
# analysiscnn_multialldays(data["feature"], data["label"])
# viewallpreproc()
# viewerfmodel(data["feature"], data["label"],  "rfr_27.pkl")
# viewerfmodel(data["feature"], data["label"],  "rfr_33.pkl")
# viewerfmodel(data["feature"], data["label"],  "rfr_44.pkl")
# viewsolemodel('north' + "42_day_cnn", 1)
# viewsolemodel('east' + "29_day_cnn", 2)
# viewsolemodel('up' + "43_day_cnn", 3)

from numpy.random import random
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# 期望输入数据尺寸: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # 返回维度为 32 的向量序列
model.add(LSTM(32, return_sequences=True))  # 返回维度为 32 的向量序列
model.add(LSTM(32))  # 返回维度为 32 的单个向量
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()
# 生成虚拟训练数据
x_train = random((1000, timesteps, data_dim))
y_train = random((1000, num_classes))

# 生成虚拟验证数据
x_val = random((100, timesteps, data_dim))
y_val = random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))




