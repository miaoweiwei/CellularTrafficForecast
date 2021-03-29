#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/12/28 22:17
@Author  : miaoweiwei
@File    : DataProcess.py
@Software: PyCharm
@Desc    : 处理原始数据
把原始数据处理之后只保留 移动网络流量的值
"""
import os
import time
import json
import datetime
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool

# 日期解析
datetime_parse = lambda x: datetime.datetime.fromtimestamp(float(x) / 1000)


def data_extraction(source_path, new_path, *args):
    """ 对原始数据进行数据提取
    :param source_path: 原始数据的文件路径
    :param new_path: 保存提取的数据的路径
    :param args: 要保存数据的列
    """
    # sms-call-internet-mi-2013-11-01.txt
    # datetime_parse = lambda x: datetime.datetime.fromtimestamp(float(x) / 1000)
    print("开始处理：", source_path)
    source_data = pd.read_csv(source_path,
                              sep='\t',  # 按tab分割
                              encoding="utf-8-sig",
                              names=['cellid', 'datetime', 'countrycode', 'smsin', 'smsout', 'callin', 'callout',
                                     'internet'],
                              # parse_dates=['datetime'],  # 要解析的数据
                              # date_parser=datetime_parse  # 使用的解析方法
                              )
    # source_data = source_data.set_index('datetime')
    # source_data['year'] = source_data.index.year
    # source_data['month'] = source_data.index.month
    # source_data['day'] = source_data.index.day
    # source_data['hour'] = source_data.index.hour
    # source_data['minute'] = source_data.index.minute
    # source_data = source_data.groupby(['year', 'month', 'day', 'hour', 'minute', 'cellid'], as_index=False).sum()
    source_data = source_data.groupby(['datetime', 'cellid'], as_index=False).sum()
    # source_data['sms'] = source_data['smsin'] + source_data['smsout']
    # source_data['calls'] = source_data['callin'] + source_data['callout']
    # temp = source_data.loc[:, ['year', 'month', 'day', 'hour', 'minute', 'cellid', 'internet', 'sms', 'calls']]
    clo_name = [col for col in args]
    temp = source_data.loc[:, ['datetime', 'cellid'] + clo_name]
    temp.to_csv(new_path, index=False, sep=',', mode='w', encoding='utf-8')
    print(os.path.split(source_path)[-1], "，处理完毕")


def multi_preprocess_data_extraction(source_dir, new_dir, *data_name, processnum=-1):
    """
    对原始数据进行批量处理
    :param data_name:
    :param processnum:
    :param source_dir:元素数据的文件夹
    :param new_dir:存放处理后的数据的文件夹
    """
    filepaths = []
    savepoaths = []
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            if os.path.splitext(filename)[-1] == '.txt':
                filepaths.append(os.path.join(dirpath, filename))
                savepoaths.append(os.path.join(new_dir, "preprocess_" + filename))
    print("file num:{0}".format(len(filepaths)))

    print("开始处理数据...")
    if processnum == -1:
        processnum = multiprocessing.cpu_count()  # 获取逻辑处理器的个数
    pool = Pool(processes=processnum)  # 创建进程池，进程池大小为3，默认 为CPU的核数(按照虚拟逻辑内核数)
    for filepath, savepath in zip(filepaths, savepoaths):
        pool.apply_async(data_extraction, args=(filepath, savepath, data_name))
    pool.close()  # 调用了close就不能再向进程池中添加任务了
    pool.join()  # 进程同步（会等待所有子进程执行完毕在继续执行后面的语句），在调用之前要先调用close方法
    print("数据处理完毕")


def internet_data_population(df, date_time, d=5):
    """
    数据中处存在缺失的部分，需要填充
    :param df: 传入一个时刻的数据，DataFrame类型的数据
    :param date_time: 当前数据的时刻
    :param d: 填充数据时要以周围多远的数据做平均填充
    :return: 返回填充好的数据，类型为 {datetime:narray}
    """
    # 首先判断数据够不够 10000 个如果不够说明存在某个位置缺少值
    if len(df) != 10000:
        # 找到缺少数据的那个cellid然后补上数据
        conut, i, cellid = 10000, 0, 1
        while i < conut:
            if df.iloc[i].cellid != cellid:  # 数据是排好顺序的
                # print("\r时刻：{0} cellid为：{1} 的数据缺失".format(datetime_parse(float(date_time)), cellid), end=' ')
                # 先用 null 标记缺少数据的位置 转成数值矩阵之后在补充
                temp_df = pd.DataFrame([[date_time, cellid, 'null']], columns=['datetime', 'cellid', 'internet'])
                df = df.append(temp_df)
                cellid += 1
                conut -= 1
                continue
            cellid += 1
            i += 1
        # 填充完毕之后记得重新排序，因为新加的都在最后面，先按照时间排序在按照cellid排序
        df = df.sort_values(by=['datetime', 'cellid'])
        # 取出Internet数据 把数据转成和网格图一样的序列
        feature = df.values[:, 2].reshape(100, 100)[::-1, :]
        # 补充缺少数据的位置
        # 把网络数据补充为缺失值周围 (d*2+1)X(d*2+1) 范围内的平均值
        indexs = np.where(feature == 'null')  # 获取要填充数据的位置
        index_arr = list(zip(*indexs))
        for row, col in index_arr:
            up = max(row - d, 0)
            down = min(row + d + 1, 100)
            left = max(col - d, 0)
            right = min(col + d + 1, 100)

            temp = feature[up:down, left:right]
            temp[temp == 'null'] = 0
            temp = temp.astype(np.float32)
            if np.count_nonzero(temp) == 0:  # count_nonzero 计算非0的个数
                feature[row, col] = 0
            else:
                feature[row, col] = np.divide(temp.sum(), np.count_nonzero(temp))  # 取周围(d*2+1)X(d*2+1) 范围内的平均值
        return date_time, feature.astype(np.float32)
    else:
        # 数据集中说明了 100X100 的编号是从下面开始的,所以在第一个维度(行)逆置
        # region_df.values[:,2] 为 internet 数据
        # [::-1,:] 表示按照第一个维度逆置
        feature = df.values[:, 2].reshape(100, 100)[::-1, :]
        return date_time, feature.astype(np.float32)


def internet_check_fill_to_matrix(file_path):
    # Check and fill and transfer matrix
    # datetime_parse = lambda x: datetime.datetime.fromtimestamp(float(x) / 1000)
    df = pd.read_csv(file_path, sep=',',
                     # encoding="utf-8-sig",
                     # parse_dates=['datetime'], date_parser=datetime_parse
                     )

    # 从 DataFrame 中取出一列数据是一个 Series
    data_time = df['datetime']  # 获取所有的时刻值
    # 输出Series中不重复的值,返回值没有排序，返回值的类型为数组
    times = sorted(data_time.unique())  # 去重然后排序
    feature_data_dic = {}
    for time_item in times:
        region_df = df[df['datetime'].isin([time_item])]
        _, feature = internet_data_population(region_df, time_item)
        feature_data_dic[time_item] = feature
    return feature_data_dic


class DataProcess(object):
    """检查数据的缺失并填充，然后把数据转成一张张图片的形式"""

    def __init__(self, datadir):
        self.data_dir = datadir
        self.file_paths = []
        self.feature_data_dic = {}
        self._processed_count = 0
        self._start_time = 0

    def load_data_paths(self, datadir=None):
        """获取数据的文件路径"""
        if datadir is not None:
            self.data_dir = datadir
        self.file_paths.clear()
        for file_name in os.listdir(self.data_dir):
            self.file_paths.append(os.path.join(self.data_dir, file_name))
        return self.file_paths

    # _xx 以单下划线开头的表示的是protected类型的变量。即保护类型只能允许其本身与子类进行访问
    # __xx 双下划线的表示的是私有类型的变量。只能允许这个类本身进行访问了，连子类也不可以访问
    # __xx__定义的是特列方法，不要自己定义这类方法
    def _data_population_callback(self, data):
        """收集处理后的数据，并计算耗时"""
        if isinstance(data, tuple):
            date_time, feature = data
            if date_time not in self.feature_data_dic:
                self.feature_data_dic[date_time] = feature
        elif isinstance(data, dict):
            # 字典合并
            self.feature_data_dic.update(data)

        self._processed_count += 1
        print("\r {}/{} 已完成 {:3.2f}% 消耗时间：{:.3}".format(self._processed_count, len(self.file_paths),
                                                        self._processed_count / len(self.file_paths) * 100,
                                                        time.time() - self._start_time), end=' ')

    def load_parse_async(self, processes=-1):
        """开始处理数据"""
        self._start_time = time.time()
        self._processed_count = 0
        if len(self.file_paths) <= 0:
            print("文件个数小于1！")
            return
        print("文件个数为：%d" % len(self.file_paths))
        if processes == -1:
            processes = os.cpu_count() * 2
        pool = Pool(processes=processes)
        for file_path in self.file_paths:
            # internet_check_fill_to_matrix执行结束后就会执行 callback。callback的参数就是internet_check_fill_to_matrix的返回值
            pool.apply_async(internet_check_fill_to_matrix, args=(file_path,), callback=self._data_population_callback)
        pool.close()
        pool.join()
        print("")
        print("加载数据总花费时间：", time.time() - self._start_time)

    def save(self, file_path):
        # 因为 np.int64 不能被序列化，所以要转成str
        # 把numpy数组序列化list，因为numpy类型不能被json处理

        # 要先排好序在进行转换
        self.feature_data_dic = dict(sorted(self.feature_data_dic.items(), key=lambda item: item[0]))  # 按时间进行排序
        self.feature_data_dic = {str(key): value.tolist() for key, value in self.feature_data_dic.items()}
        with open(file_path, 'w') as f:
            json.dump(self.feature_data_dic, f)  # 把python对象保存成文件
        print("保存到文件完成")


if __name__ == '__main__':
    # 米兰市的电信活动
    milandatasetdir = r"F:\移动流量数据\SourceData\Milan"
    milanpredatadir = r"D:\Myproject\Python\Datasets\MobileFlowData\PreprocessingData\Milan"
    # 特伦蒂诺省的电信活动
    # trentinodatasetdir = r"D:\Myproject\Python\Datasets\MobileFlowData\SourceData\Trentino"
    # trentinopredatadir = r"D:\Myproject\Python\Datasets\MobileFlowData\PreprocessingData\Trentino"

    # 提取网络数据并进行转存
    multi_preprocess_data_extraction(milandatasetdir, milanpredatadir, 'internet')

    # 对网络数据缺失的数据进行填充，并转存成一帧一帧的数据
    process = DataProcess(milanpredatadir)
    data_paths = process.load_data_paths()
    print("文件个数：%d" % len(data_paths))

    process.load_parse_async()
    # 共62天 每天144个数据
    print("数据的总数：%d" % len(process.feature_data_dic))

    feature_dir = r"D:\Myproject\Python\Datasets\MobileFlowData\PreprocessingData"
    milan_internet_data_path = os.path.join(feature_dir, "traffic_data.txt")
    process.save(milan_internet_data_path)
