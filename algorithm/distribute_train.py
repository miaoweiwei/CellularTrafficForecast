#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/6/17 14:58
@Author  : miaoweiwei
@File    : distribute_train.py
@Software: PyCharm
@Desc    : 
"""
import os

import tensorflow as tf
from tensorflow import keras

from algorithm import data_provider

"""
GPU的相关设置
"""
# 显示GPU日志
tf.debugging.set_log_device_placement(True)
# 设置自动分配计算到指定的设备上
tf.config.set_soft_device_placement(True)
# 获取物理GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
print("物理GPU数量：", len(gpus))
for gpu in gpus:
    tf.config.experimental.set_visible_devices(gpu, 'GPU')  # 设置对本进程可见的GPU设备
    tf.config.experimental.set_memory_growth(gpu, True)  # 设置GPU内存自增长

logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print("逻辑GPU数量：", len(logical_gpus))

"""
数据预处理
"""
seq_length = 17  # 序列的长度 输入的长度+输出的长度
d = 7
data_path = "D:/Myproject/Python/Datasets/MobileFlowData/PreprocessingData/milan_feature.txt"
batch_size = 1024 * len(logical_gpus)

# 构建数据集
provider = data_provider.DataProvider(data_path, time_slice=seq_length, relevance_distance=d)
dataset = data_provider.Dateset(provider)
# 在模型里使用了批归一化，数据就不需要在做归一化了
dataset.provider(valid_size=0.1, test_size=0.1, israndom=False, isnorm=True)
train_dataset, valid_dataset, test_dataset = dataset.get_dataset(batch_size,
                                                                 train_prefetch=32,
                                                                 valid_prefetch=32,
                                                                 test_prefetch=32)
for inputs, outputs in test_dataset.take(2):
    print(type(inputs), type(outputs))
    print(inputs.shape, outputs.shape)

"""
加载模型
"""
logdir = os.path.join('tcn2d_3dconv')
if not os.path.exists(logdir):
    os.makedirs(logdir)
output_model_file = os.path.join(logdir, 'tcn2d_3dconv.h5')
# 这里的路径要使用 os.path.join 包装一下，不然会报错
callbacks = [
    keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_images=True),
    # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto',
    #                                   min_delta=0.0001, cooldown=0, min_lr=0),
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True, save_weights_only=False),  # 保存模型和权重
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]
