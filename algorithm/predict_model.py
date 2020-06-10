#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/5/1 20:16
@Author  : miaoweiwei
@File    : predict_model.py
@Software: PyCharm
@Desc    : 
"""
import os
from tensorflow import keras
import matplotlib.pyplot as plt
from algorithm import data_provider, models
from sklearn.metrics import mean_squared_error

seq_length = 17  # 序列的长度 输入的长度+输出的长度
d = 7
data_path = "D:/Myproject/Python/Datasets/MobileFlowData/PreprocessingData/milan_feature.txt"
batch_size = 1024

provider = data_provider.DataProvider(data_path, time_slice=seq_length, relevance_distance=d)
train_data, valid_data, test_data = provider.provider(valid_size=0.3, test_size=0.3, israndom=False,
                                                      norm_func='log10p', isnorm=False)

logdir = os.path.join('Tcn_Model_Files')
if not os.path.exists(logdir):
    os.makedirs(logdir)
output_model_file = os.path.join(logdir, 'tcn2d_model.h5')
if os.path.isfile(output_model_file):
    model = keras.models.load_model(output_model_file)
else:
    model = models.tcn_model(time_slice=seq_length - 1, relevance_distance=d)

x_, y_ = provider.get_test_data(test_data, 550, 1000, 1050)

x = provider.data_normalized(x_, 'log10p')
pre = model.predict(x)
pre_ = provider.data_anti_normalized(pre, 'log10p')

mse = mean_squared_error(pre_, y_)

plt.plot(y_, marker='.', mec='r', mfc='w')
plt.plot(pre_, marker='o', ms=10)
plt.show()
