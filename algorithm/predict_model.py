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

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow import keras

from algorithm import data_provider

# tf.debugging.set_log_device_placement(True)# 显示GPU的一些信息
gpus = tf.config.experimental.list_physical_devices("GPU")
print("物理GPU的数量：", len(gpus))
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')  # 设置对本进程可见的GPU设备
# 设置GPU内存自增长必须放到程序刚开始的地方，否则会报错
tf.config.experimental.set_memory_growth(gpus[0], True)

logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print("逻辑GPU的数量：", len(logical_gpus))

seq_length = 17  # 序列的长度 输入的长度+输出的长度
d = 7
data_path = "D:\Myproject\Python\Datasets\MobileFlowData\PreprocessingData\milan_feature.txt"
batch_size = 1600 * 2

random_seed = 666

# 固定随机数种子
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
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

x_, y_ = provider.get_test_data(provider.test_data, 500, 300, 350)
x = provider.data_normalized(x_, 'log1p')

pre = model.predict(x)
pre_ = provider.data_anti_normalized(pre, 'expm1')
mse_ = mean_squared_error(pre_, y_)

y = provider.data_normalized(y_, 'log1p')
mse = mean_squared_error(pre, y)

plt.plot(y_, mec='r', label="ground truth")  # 真实的
plt.plot(pre_, linestyle='--', ms=10, label="2DTCN+3DConv")
plt.legend()
plt.show()
print(mse_)

plt.plot(y, mec='r', label="ground truth")  # 真实的
plt.plot(pre, linestyle='--', ms=10, label="2DTCN+3DConv")
plt.legend()
plt.title("Normalized")
plt.show()
print(mse)
