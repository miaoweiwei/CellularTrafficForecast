#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/5/1 14:36
@Author  : miaoweiwei
@File    : data_test.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
from algorithm import data_provider

seq_length = 17  # 序列的长度 输入的长度+输出的长度
d = 7
data_path = "D:/Myproject/Python/Datasets/MobileFlowData/PreprocessingData/milan_feature.txt"
batch_size = 1024

provider = data_provider.DataProvider(data_path, time_slice=seq_length, relevance_distance=d)
train_data, valid_data, test_data = provider.provider(valid_size=0.3, test_size=0.3, israndom=False,
                                                      norm_func='log10p', isnorm=False)
del train_data, valid_data
x, y = provider.get_test_data(test_data, 550, 0, 50)

print(x.shape, y.shape)
