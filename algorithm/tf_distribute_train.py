#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/6/10 10:22
@Author  : miaoweiwei
@File    : tf_distribute_train.py
@Software: PyCharm
@Desc    : 
"""

import os
import sys
import math
import tqdm
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from algorithm import data_provider, models

tf.debugging.set_log_device_placement(True)  # 显示GPU的一些信息
gpus = tf.config.experimental.list_physical_devices("GPU")
print("物理GPU的数量：", len(gpus))
# 设置GPU内存自增长必须放到程序刚开始的地方，否则会报错
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print("逻辑GPU的数量：", len(logical_gpus))

# seq_length = 13  # 序列的长度 输入的长度+输出的长度
# d = 5
seq_length = 17  # 序列的长度 输入的长度+输出的长度
d = 7
data_path = "D:/Myproject/Python/Datasets/MobileFlowData/PreprocessingData/milan_feature.txt"
batch_size = 1024 * len(logical_gpus)

provider = data_provider.DataProvider(data_path, time_slice=seq_length, relevance_distance=d)
# 在模型里使用了批归一化，数据就不需要在做归一化了
train_data, valid_data, test_data = provider.provider(valid_size=0.1, test_size=0.1, israndom=False, isnorm=True)

# 构建数据集
train_dataset = tf.data.Dataset.from_generator(
    lambda: provider.correlation_split_generate(data=train_data), output_types=tf.float64)
# 验证集不需要打乱
valid_dataset = tf.data.Dataset.from_generator(
    lambda: provider.correlation_split_generate(data=valid_data, shuffle=False), output_types=tf.float64)
# 测试集不需要打乱
test_dataset = tf.data.Dataset.from_generator(
    lambda: provider.correlation_split_generate(data=test_data, shuffle=False), output_types=tf.float64)

for flow in train_dataset.take(1):
    print(flow.shape)

# 把 train_dataset 里的每一个小的序列, 形状[13, 11, 11]
# 使用 split_input_target 函数处理
# 生成 input 和 output, 形状分别为 [12,11,11], [1,] 取最后一帧的中间的数据为输出
train_dataset = train_dataset.map(lambda x: provider.split_input_target(x), num_parallel_calls=os.cpu_count() * 2)
# num_parallel_calls值为“ tf.data.experimental.AUTOTUNE”，那么将根据可用的CPU动态设置并行调用的次数。
valid_dataset = valid_dataset.map(lambda x: provider.split_input_target(x), num_parallel_calls=os.cpu_count() * 2)
test_dataset = test_dataset.map(lambda x: provider.split_input_target(x), num_parallel_calls=os.cpu_count() * 2)

# correlation_split_generate 函数已经进行了数据的打乱
# train_dataset = train_dataset.shuffle(buffer_size=batch_size)
# drop_remainder=False最后的不够一个batch_size 也要
train_dataset = train_dataset.batch(batch_size, drop_remainder=False).repeat().prefetch(32)  # repeat 参数为空表示意思是重复无数遍
valid_dataset = valid_dataset.batch(batch_size, drop_remainder=False).repeat().prefetch(32)
test_dataset = test_dataset.batch(batch_size=batch_size).repeat().prefetch(32)

for inputs, outputs in train_dataset.take(2):
    print(type(inputs), type(outputs))
    print(inputs.shape, outputs.shape)

# model = models.stn_model(time_slice=seq_length - 1, relevance_distance=d)
# 输入是形状为6 × 15 × 15 的矩阵
# model = models.stfm_model(time_slice=seq_length - 1, relevance_distance=d)
# model = models.stfm(time_slice=seq_length - 1, relevance_distance=d)

# logdir = os.path.join('Tcn_Model_Files')
logdir = os.path.join('Tcn_Model_Files')
if not os.path.exists(logdir):
    os.makedirs(logdir)
output_model_file = os.path.join(logdir, 'tcn2d_model.h5')

# 这里的路径要使用 os.path.join 包装一下，不然会报错
callbacks = [
    keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_images=True),
    # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto',
    #                                   min_delta=0.0001, cooldown=0, min_lr=0),
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True, save_weights_only=False),  # 保存模型和权重
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    if os.path.isfile(output_model_file):
        model = keras.models.load_model(output_model_file)
    else:
        model = models.tcn_model(time_slice=seq_length - 1, relevance_distance=d)
    # loss='mean_squared_error',

    # keras.losses.logcosh
    # Log-cosh Loss
    # **优点：**当x值较小 时，log(cosh(x))约等于(x ** 2) / 2；
    # 当x值较大时，约等于abs(x) - log(2)。这意味着’logcosh’的作用大部分类似于均方误差，
    # 但不会受到偶然误差预测的强烈影响。它具有Huber损失的所有优点，并且它在各处都是可区分的，
    # 与Huber损失不同。

    # 均方对数误差 keras.losses.mean_squared_logarithmic_error
    # 优点惩罚预测大于过预测，就是当预测值小于真实值
    # 当数据中存在少量的值和真实值差值较大的时候，
    # 使用这个函数能够减少这些值对整体误差的影响

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(lr=0.001, decay=0.1, epsilon=1e-8),
                  # optimizer='SGD',
                  # metrics=['accuracy']
                  )
epochs = 20
steps_per_epoch = math.ceil((len(train_data) - seq_length + 1) * 100 * 100 / batch_size)
validation_steps = math.ceil((len(valid_data) - seq_length + 1) * 100 * 100 / batch_size)
history = model.fit_generator(train_dataset,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=valid_dataset,
                              validation_steps=validation_steps,
                              epochs=epochs,
                              callbacks=callbacks)
