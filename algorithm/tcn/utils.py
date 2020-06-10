#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/4/21 13:12
@Author  : miaoweiwei
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""


def is_power_of_two(num):
    return num != 0 and ((num & (num - 1)) == 0)


def adjust_dilations(dilations):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations
