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
data_path = "D:\Myproject\Python\Datasets\MobileFlowData\PreprocessingData\milan_feature.txt"
batch_size = 1600 * 2

# 构建数据集
provider = data_provider.DataProvider(data_path, time_slice=seq_length, relevance_distance=d)
dataset = data_provider.Dateset(provider)
# 在模型里使用了批归一化，数据就不需要在做归一化了
dataset.ceate_data(valid_size=0.1, test_size=0.1, israndom=True, isnorm=False)
train_dataset, valid_dataset, test_dataset = dataset.get_dataset(batch_size,
                                                                 train_prefetch=4,
                                                                 valid_prefetch=4,
                                                                 test_prefetch=4)
for inputs, outputs in test_dataset.take(2):
    print(type(inputs), type(outputs))
    print(inputs.shape, outputs.shape)

logdir = os.path.join('conv3d_tcn2d')
output_model_file = os.path.join(logdir, 'conv3d_tcn2d.h5')
if os.path.isfile(output_model_file):
    model = keras.models.load_model(output_model_file)
else:
    print("模型：", output_model_file, "，不存在")
    exit(0)
model.summary()

x_, y_ = provider.get_test_data(provider.test_data, 500, 300, 400)
x = provider.data_normalized(x_, 'log1p')

pre = model.predict(x)
pre_ = provider.data_anti_normalized(pre, 'expm1')

mse = mean_squared_error(pre_, y_)

plt.plot(y_, marker='.', mec='r', mfc='w')
plt.plot(pre_, marker='*', ms=10)
plt.show()
print(mse)
