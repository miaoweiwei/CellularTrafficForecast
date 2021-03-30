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
data_path = r"D:\Myproject\Python\Datasets\MobileFlowData\PreprocessingData\traffic_data.txt"
batch_size = 1600 * 2

# random_seed = 666
#
# # 固定随机数种子
# np.random.seed(random_seed)
# tf.random.set_seed(random_seed)
# 构建数据集
provider = data_provider.DataProvider(data_path, time_slice=seq_length, relevance_distance=d)
dataset = data_provider.Dateset(provider)
# 在模型里使用了批归一化，数据就不需要在做归一化了,这里是预测，所以就不用把所有的数据都进行归一化了，在后面预测的时在进行处理
dataset.ceate_data(valid_size=0.1, test_size=0.1, israndom=True, isnorm=False)
train_dataset, valid_dataset, test_dataset = dataset.get_dataset(batch_size,
                                                                 train_prefetch=4,
                                                                 valid_prefetch=4,
                                                                 test_prefetch=32)
for inputs, outputs in test_dataset.take(2):
    print(type(inputs), type(outputs))
    print(inputs.shape, outputs.shape)

model_path = r"E:\毕业季\毕业设计\毕业设计\模型文件\tcn2d_model_BN_20200507"
mode_name = "tcn2d_model.h5"
output_model_file = os.path.join(model_path, mode_name)
if os.path.isfile(output_model_file):
    model = keras.models.load_model(output_model_file)
    model.summary()
else:
    print("模型：", mode_name, "，不存在")
    exit(0)

# 选择数据进行预测
# 真实数据
x_, y_ = provider.get_test_data(provider.test_data, 2500, 300, 400)
# x = provider.data_normalized(x_, 'log1p')
# y = provider.data_normalized(y_, 'log1p')

# 预测数据
pre = model.predict(x_)
# 反归一化得到真实值
# pre_ = provider.data_anti_normalized(pre, 'expm1')

# 计算误差 真实值的均方误差
mse_ = mean_squared_error(pre, y_)
print(mse_)
# 归一化后的均方误差
# mse = mean_squared_error(pre, y)
# print(mse)

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.plot(y_, mec='r', label="ground truth")  # 真实的
plt.plot(pre, linestyle='--', ms=10, label="2DTCN")
plt.legend()
# plt.title("no normalized")
plt.show()

# plt.plot(y, mec='r', label="ground truth")
# plt.plot(pre, linestyle='--', ms=10, label="2DTCN")
# plt.legend()
# plt.title("log1p normalized")
# plt.show()
