#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/4/8 21:23
@Author  : miaoweiwei
@File    : data_provider.py
@Software: PyCharm
@Desc    : 
"""
import json
import math
import os
import time

import numpy as np
import tensorflow as tf


class DataProvider(object):
    def __init__(self, data_path, time_slice=13, relevance_distance=5):
        """
        :param data_path:数据集的文件路径
        :param time_slice: 序列的长度包括输入序列的长度+输出序列的长度
        :param relevance_distance: 数据之间的相关性距离
        """
        self.data_path = data_path
        self.time_slice = time_slice
        self.relevance_distance = relevance_distance

        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def provider(self, valid_size=0.2, test_size=0.2, israndom=True, norm_func='z_scores', isnorm=True):
        """ 数据的获取
        :param valid_size: 验证集的比例
        :param test_size: 测试集的比例
        :param israndom: 划分数据集时是否随机打乱，
        :param norm_func: 归一化的方法
        :param isnorm: 是否归一化
        :return:
        """
        features = self._load_data()
        if isnorm:
            features = DataProvider.data_normalized(features, norm_func=norm_func)
        features = self._create_seq(features)
        self.train_data, self.valid_data, self.test_data = DataProvider.data_split(features, valid_size, test_size,
                                                                                   israndom=israndom)
        # 删除不用的变量一遍节省内存
        del features
        return self.train_data, self.valid_data, self.test_data

    @staticmethod
    def data_normalized(data, norm_func='z_scores', is_prediction=False, mean=0, std=1):
        if norm_func == 'log1p':
            return np.log1p(data)
        if norm_func == 'log10p':
            return np.log10(data + 1)
        if norm_func == 'arctan':
            return np.atan(data)
        if norm_func == 'z_scores':
            if not is_prediction:
                mean = np.mean(data)
                std = np.std(data)
                with open("mean_std.txt", "w") as f:
                    f.write("mean:{0}\n".format(mean))
                    f.write("std:{0}\n".format(std))
                return (data - mean) / std
            else:  # 当前是预测
                return (data - mean) / std

    @staticmethod
    def data_anti_normalized(data, norm_func='z_scores', mean=0, std=1):
        if norm_func == 'log1p':
            return np.expm1(data)  # 为np.log1p(data)的反向操作
        if norm_func == 'log10p':
            return np.power(10, data) - 1
        if norm_func == 'tan':
            return np.tan(data)
        if norm_func == 'z_scores':
            return data * std + mean

    def _load_data(self):
        start_time = time.time()
        # 使用Json加载数据集，并解析
        with open(self.data_path) as f:
            feature_dic = json.load(f)

        feature_list = list(feature_dic.items())
        times, features = zip(*feature_list)
        # 删除不需要的数据，以便节约内存
        del feature_dic, feature_list, times
        # 流量数据处理成 [8928,100,100] 的numpy类型的数据，
        # 8928 = 62 *144，共62天，每天十分钟取一次数据共144个数据
        features = np.array(features)
        m, s = divmod(time.time() - start_time, 60)
        h, m = divmod(m, 60)
        print("文件加载耗时: {:0>2d}:{:0>2d}:{:0>2d}".format(int(h), int(m), int(s)))
        return features

    def _create_seq(self, flow_matrix):
        """ 数据序列化，
        :param flow_matrix: 数据集矩阵
        :return:返回的数据形状为 [len(flow_matrix) - self.time_slice + 1, self.time_slice, flow_matrix.shape[1], flow_matrix.shape[2]]
        """
        start_time = time.time()
        # 生成序列的总数为 len(flow_matrix) - self.time_slice + 1
        # 形状为 [len(flow_matrix) - self.time_slice + 1, self.time_slice, flow_matrix.shape[1], flow_matrix.shape[2]]
        result = np.array(
            [flow_matrix[i:i + self.time_slice] for i in range(flow_matrix.shape[0] - self.time_slice + 1)])

        m, s = divmod(time.time() - start_time, 60)
        h, m = divmod(m, 60)
        print("数据序列化耗时: {:0>2d}:{:0>2d}:{:0>2d}".format(int(h), int(m), int(s)))
        return result

    @staticmethod
    def data_split(data, valid_size=0.2, test_size=0.2, israndom=True):
        """ 划分数据集
        :param data: 源数据集
        :param valid_size: 验证集所占比例
        :param test_size: 测试集所占比例
        :param israndom: 是否随机划分数据集,不随机划分就是按照顺序划分
        :return: 训练集， 验证集， 测试集
        """
        start_time = time.time()
        data_length = len(data)
        valid_length = int(valid_size * data_length)
        test_length = int(test_size * data_length)
        train_length = data_length - valid_length - test_length
        if israndom:
            data = np.random.permutation(data)  # 打乱数据
            # 划分数据
            valid_data = data[0:valid_length]
            test_data = data[valid_length:valid_length + test_length]
            train_data = data[valid_length + test_length:]
        else:
            train_data, test_data = np.random.permutation(data[0:train_length + valid_length]), np.random.permutation(
                data[train_length + valid_length:])
            train_data, valid_data = train_data[0:train_length], train_data[train_length:]

        m, s = divmod(time.time() - start_time, 60)
        h, m = divmod(m, 60)
        print("数据划分耗时: {:0>2d}:{:0>2d}:{:0>2d}".format(int(h), int(m), int(s)))
        return train_data, valid_data, test_data

    def _correlation_split(self, flow_matrix, padding="same"):
        """ 对一个序列的数据进行处理,切割
        :param flow_matrix: 一个序列的数据
        :param padding: 是否需要填充
        :return:返回处理后的数据
        """
        assert len(flow_matrix.shape) == 3, "输入数据的维度不是3,输入数据的形状应为[channel, 高, 宽]"
        padding = padding.lower()
        assert padding in ("same", "valid"), "padding 的参数 {0} 不存在 (‘same’, ‘valid’)中".format(padding)

        filter_size = (2 * self.relevance_distance + 1, 2 * self.relevance_distance + 1)
        channel, h, w = flow_matrix.shape
        result = None
        if padding == 'same':  # 需要填充
            # 每一张feature的高和框维度上要填充的维度
            p = [int((size - 1) / 2) for size in filter_size]
            filling = [[0, 0], p, p]
            # 每一个feature map周围补充0
            # numpy填充
            flow_matrix = np.lib.pad(flow_matrix,
                                     filling,
                                     'constant',
                                     constant_values=0.)
            # Tensor填充
            # flow_matrix = tf.pad(flow_matrix,
            #                      tf.constant(filling),
            #                      "CONSTANT",
            #                      constant_values=0)
            # 输出结果的 numpy 数组
            result = np.zeros((h * w, channel, filter_size[0], filter_size[1]))
        else:  # 不需要填充
            # 输出结果的 numpy 数组
            result = np.zeros(((h - filter_size[0] + 1) * (w - filter_size[1] + 1),
                               channel, filter_size[0], filter_size[1]))

        """
        在第二歌维度上逆置，第一个维度是时间，第二和第三个维度是feature的高和宽，
        feature原来是
            9901 9902 9903 ... 9998 9999 10000
                            .
            201  202  203  ... 298  299  300
            101  102  103  ... 198  199  200
            1    2    3    ... 98   99   100
        在第二歌维度逆置后变成
            1    2    3    ... 98   99   100
            101  102  103  ... 198  199  200
            201  202  203  ... 298  299  300
                            .
            9901 9902 9903 ... 9998 9999 10000
        这样生成的结果的顺序就是从1到10000
        """
        flow_matrix = flow_matrix[:, ::-1]
        index = 0
        for i in range(h - (filter_size[0] - 1)):
            for j in range(w - (filter_size[1] - 1)):
                result[index, :, :, :] = flow_matrix[:, i:i + filter_size[0], j:j + filter_size[1]]
                index += 1
        return result

    def correlation_split_generate(self, data, shuffle=True, padding="same"):
        """ 使用生成器对数据进行处理
        :param data: 序列的数据
        :param shuffle: 输出在同一帧上的数据是否打乱
        :param padding: 是否需要填充
        """
        while True:
            for flow in data:
                # 对一个序列的数据进行填充分割， flow的形状为[time_slice,100,100]
                feature_map = self._correlation_split(flow, padding)
                if shuffle:  # 打乱数据，这里打乱的是在一帧上面打乱数据
                    feature_map = np.random.permutation(feature_map)
                for temp in feature_map:
                    yield temp  # 形状为 [time_slice, relevance_distance*2+1, relevance_distance*2+1],如[13,11,11]

    def split_input_target(self, flow, is_prediction=False):
        """
        分离输入数据和标签
        """
        # 输出为最后一张图的中心位置的值
        # 把数据集的输入reshape成 (12, 11, 11, 1)
        width = height = 2 * self.relevance_distance + 1
        if is_prediction:
            seq = flow[0:-1].reshape((self.time_slice - 1, width, height, 1))
        else:
            seq = tf.reshape(flow[0:-1], shape=(self.time_slice - 1, width, height, 1))
        return seq, flow[-1, self.relevance_distance, self.relevance_distance]

    def get_test_data(self, data, cell_id, start, end):
        """ 获取测试的数据，用于数据的预测
        :param data: 序列的数据
        :param cell_id: 要预测的方格的id
        :param start: 开始的序列额编号
        :param end: 结束的序列额编号
        """
        start_time = time.time()
        x = []
        y = []
        i = start
        while i < len(data) and i < end:
            feature_map = self._correlation_split(data[i], padding="same")
            x_, y_ = self.split_input_target(feature_map[cell_id - 1], is_prediction=True)
            del feature_map
            x.append(x_)
            y.append(y_)
            i += 1
        m, s = divmod(time.time() - start_time, 60)
        h, m = divmod(m, 60)
        print("获取测试的数据耗时: {:0>2d}:{:0>2d}:{:0>2d}".format(int(h), int(m), int(s)))
        return np.array(x), np.array(y)


class Dateset(object):
    def __init__(self, provider):
        self.provider = provider

    def ceate_data(self, valid_size=0.2, test_size=0.2, israndom=True, norm_func='z_scores', isnorm=True):
        train_data, valid_data, test_data = self.provider.provider(valid_size=valid_size,
                                                                   test_size=test_size,
                                                                   israndom=israndom,
                                                                   norm_func=norm_func,
                                                                   isnorm=isnorm)
        return train_data, valid_data, test_data

    def _creat_dataset(self, data, batch_size, drop_remainder=True, prefetch_size=32):
        self.batch_size = batch_size
        # 构建数据集
        dataset = tf.data.Dataset.from_generator(lambda: self.provider.correlation_split_generate(data=data),
                                                 output_types=tf.float64)
        # 把 train_dataset 里的每一个小的序列, 形状[13, 11, 11]
        # 使用 split_input_target 函数处理
        # 生成 input 和 output, 形状分别为 [12,11,11], [1,] 取最后一帧的中间的数据为输出
        # num_parallel_calls值为“ tf.data.experimental.AUTOTUNE”，那么将根据可用的CPU动态设置并行调用的次数。
        dataset = dataset.map(lambda x: self.provider.split_input_target(x), num_parallel_calls=os.cpu_count() * 2)
        # correlation_split_generate 函数已经进行了数据的打乱
        # train_dataset = train_dataset.shuffle(buffer_size=batch_size)
        # drop_remainder=False最后的不够一个batch_size 也要
        # repeat 参数为空表示意思是重复无数遍
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).repeat().prefetch(prefetch_size)
        return dataset

    def get_dataset(self, batch_size, train_prefetch=32, valid_prefetch=32, test_prefetch=32):
        train_dataset = self._creat_dataset(self.provider.train_data, batch_size, prefetch_size=train_prefetch)
        valid_dataset = self._creat_dataset(self.provider.valid_data, batch_size, prefetch_size=valid_prefetch)
        test_dataset = self._creat_dataset(self.provider.test_data, batch_size, prefetch_size=test_prefetch)
        return train_dataset, valid_dataset, test_dataset

    def _step(self, data):
        """计算每一个轮次有多个batch_Size"""
        return math.ceil((len(data) - self.provider.time_slice + 1) * 100 * 100 / self.batch_size)

    @property
    def train_step_per_epoch(self):
        return self._step(self.provider.train_data)

    @property
    def valid_step_per_epoch(self):
        return self._step(self.provider.valid_data)

    @property
    def test_step_per_epoch(self):
        return self._step(self.provider.test_data)
