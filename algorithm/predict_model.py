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
data_path = "E:/miao/dataset/milan_feature.txt"
batch_size = 1600 * 2

provider = data_provider.DataProvider(data_path, time_slice=seq_length, relevance_distance=d)
train_data, valid_data, test_data = provider.provider(valid_size=0.1, test_size=0.1, israndom=False, isnorm=False)

logdir = os.path.join('tcn2d_model')
output_model_file = os.path.join(logdir, 'tcn2d_model.h5')
if os.path.isfile(output_model_file):
    model = keras.models.load_model(output_model_file)
else:
    print("模型：", output_model_file, "，不存在")
    exit(0)
model.summary()

x_, y_ = provider.get_test_data(test_data, 500, 400, 450)
x = provider.data_normalized(x_, 'log1p')

pre = model.predict(x)
pre_ = provider.data_anti_normalized(pre, 'expm1')

mse = mean_squared_error(pre_, y_)

plt.plot(y_, marker='.', mec='r', mfc='w')
plt.plot(pre_, marker='*', ms=10)
plt.show()
print(mse)
