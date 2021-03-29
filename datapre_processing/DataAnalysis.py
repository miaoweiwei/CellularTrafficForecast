#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/1/2 14:45
@Author  : miaoweiwei
@File    : DataAnalysis.py
@Software: PyCharm
@Desc    : 对原始数据预处理后的数据进行特征分析
"""
import os
import time
import math
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("ticks")
sns.set_context("paper")
# 日期解析
DatetimeParse = lambda x: datetime.datetime.fromtimestamp(float(x) / 1000)


def date_parser(x):
    return datetime.datetime.fromtimestamp(float(x) / 1000)


def day_folw_trend(files=[]):
    file_sum = pd.DataFrame({})
    for filepath in files:
        file_temp = pd.read_csv(
            filepath,
            sep='\t',  # 按tab分割
            encoding="utf-8-sig",
            names=[
                'cellid', 'datetime', 'countrycode', 'smsin', 'smsout', 'callin',
                'callout', 'internet'
            ],
            parse_dates=['datetime'],  # 要解析的数据
            date_parser=DatetimeParse  # 使用的解析方法
        )
        file_temp = file_temp.set_index('datetime')
        file_temp['hour'] = file_temp.index.hour
        file_temp['weekday'] = file_temp.index.weekday
        file_temp = file_temp.groupby(['hour', 'weekday'], as_index=False).sum()
        file_sum = file_sum.append(file_temp)
    file_sum = file_sum.groupby(['hour', 'weekday'], as_index=False).sum()
    file_day = file_sum[file_sum.weekday == 1]
    file_day.head()
    # z - score标准化
    # z-score标准化方法适用于属性A的最大值和最小值未知的情况，或有超出取值范围的离群数据的情况。
    # 该种归一化方式要求原始数据的分布可以近似为高斯分布，否则归一化的效果会变得很糟糕。
    sliceSum_z = (file_day - file_day.mean()) / file_day.std()  # (file_sum - 均值)/标准差 z-score 标准化

    fig_width_pt = 345  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches 转换pt到英寸
    golden_mean = (math.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio 黄金分割比例
    fig_width = fig_width_pt * inches_per_pt  # width in inches 宽度（英寸）
    fig_height = fig_width * golden_mean * 2  # height in inches 高度（英寸）
    fig_size = [fig_width, fig_height]
    # Behaviour plot
    types = ['smsin', 'callin', 'internet']
    f, axs = plt.subplots(len(types), sharex=True, sharey=True, figsize=fig_size)

    timeSlice = [i for i in range(7, 25)]
    timeSlice = timeSlice + [i for i in range(1, 7)]
    for i, p in enumerate(types):
        # plt.xticks(np.arange(24, step=1))  # 第一天的7点到第二天的6点
        plt.xticks(timeSlice)  # 第一天的7点到第二天的6点
        axs[i].plot(sliceSum_z[p].values, label=p)
        axs[i].legend(loc='upper center')  # 每一个图的图例显示在图的上方中心的位置
        sns.despine()
        f.text(0, 0.5, "Number of events", rotation="vertical", va="center")
        plt.xlabel("Hour (in a day,2013-11-05)")
        plt.savefig('day_folw_trend.pdf', format='pdf', dpi=330, bbox_inches='tight')
        plt.show()


def week_folw_trend(files=[]):
    if len(files) != 7:
        print("日期不为7天")
    print("开始处理")
    file_sum = pd.DataFrame({})
    for filepath in files:
        file_temp = pd.read_csv(filepath, sep=',',
                                encoding="utf-8-sig",
                                parse_dates=['datetime'],
                                date_parser=DatetimeParse)
        file_temp = file_temp.set_index('datetime')
        file_temp['hour'] = file_temp.index.hour
        file_temp['weekday'] = file_temp.index.weekday
        file_temp = file_temp.groupby(['hour', 'weekday', 'cellid'], as_index=False).sum()
        file_sum = file_sum.append(file_temp)
        print("文件 {0} 加载完成".format(filepath))

    file_sum = file_sum.groupby(['weekday', 'hour'], as_index=False).sum()
    file_sum['idx'] = file_sum['hour'] + (file_sum['weekday'] * 24)
    file_sum = file_sum.sort_values(by='idx')
    file_sum.head()
    # Z-score
    sliceSum_z = (file_sum - file_sum.mean()) / file_sum.std()  # (sliceSum_city - 均值)/标准差

    fig_width_pt = 345  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches 转换pt到英寸
    golden_mean = (math.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio 黄金分割比例
    fig_width = fig_width_pt * inches_per_pt  # width in inches 宽度（英寸）
    fig_height = fig_width * golden_mean * 2  # height in inches 高度（英寸）
    fig_size = [fig_width, fig_height]
    # Behaviour plot
    types = ['smsin', 'callin', 'internet']
    f, axs = plt.subplots(len(types), sharex=True, sharey=True, figsize=fig_size)
    for i, p in enumerate(types):
        plt.xticks(np.arange(168, step=10))  # 一周 168小时
        axs[i].plot(sliceSum_z[p], label=p)
        axs[i].legend(loc='upper center')  # 每一个图的图例显示在图的上方中心的位置
        sns.despine()
    f.text(0, 0.5, "Number of events", rotation="vertical", va="center")
    plt.xlabel("Hour (in a week)")
    plt.savefig('week_folw_trend.pdf', format='pdf', dpi=330, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # milanpredatadir = r"D:\Myproject\Python\Datasets\MobileFlowData\PreprocessingData\Milan"
    data_path = r"D:\Myproject\Python\Datasets\MobileFlowData\SourceData\Milan"
    # 2013-11-04  2013-11-10 总共一周 ，从周一到周日
    file_name = 'sms-call-internet-mi-2013-11-{0}.txt'
    files = []
    for i in range(4, 11):
        files.append(os.path.join(data_path, file_name.format(str(i).zfill(2))))
    # week_folw_trend(files)
    day_folw_trend(files[:2])
