#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/3/9 20:45
@Author  : miaoweiwei
@File    : models.py
@Software: PyCharm
@Desc    : algorithm
"""

from tensorflow import keras
from algorithm.tcn import TCN


def _get_inputs(time_slice=12, relevance_distance=5):
    """ 获取模型的输入
    :param time_slice: 时间片的长度，也就是序列的长度
    :param relevance_distance: 相关性距离
    :return:
    """
    width = height = 2 * relevance_distance + 1
    # 通道：1，长：11 宽：11 序列长度：12
    inputs = keras.layers.Input(shape=(time_slice, height, width, 1))
    return inputs


def stn_model(time_slice=12, relevance_distance=5):
    """
    论文 Long-Term Mobile Traffic Forecasting Using Deep Spatio-Temporal Neural Networks
    论文地址：https://arxiv.org/pdf/1712.08083v1.pdf
    三维卷积 + 卷积的LSTM 的融合
    :param time_slice: 序列的时间片个，就是序列的长度
    :param relevance_distance: 相关性距离
    :return:
    """
    inputs = _get_inputs(time_slice=time_slice, relevance_distance=relevance_distance)
    conv1 = keras.layers.Convolution3D(3,
                                       kernel_size=(3, 3, 3),
                                       activation='relu',
                                       padding='same')(inputs)
    lstm1 = keras.layers.ConvLSTM2D(3,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    return_sequences=True)(inputs)
    x1 = keras.layers.concatenate([conv1, lstm1], axis=4)  # 第四个维度上相加

    conv2 = keras.layers.Convolution3D(6,
                                       kernel_size=(3, 3, 3),
                                       activation='relu',
                                       padding='same')(x1)
    lstm2 = keras.layers.ConvLSTM2D(6,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    return_sequences=True)(x1)
    x2 = keras.layers.concatenate([conv2, lstm2], axis=4)  # 第四个维度上相加

    conv3 = keras.layers.Convolution3D(12,
                                       kernel_size=(3, 3, 3),
                                       activation='relu',
                                       padding='same')(x2)
    lstm3 = keras.layers.ConvLSTM2D(12,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    return_sequences=True)(x2)
    x3 = keras.layers.concatenate([conv3, lstm3], axis=4)  # 第四个维度上相加

    x = keras.layers.Flatten()(x3)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def _conv_block(inputs, kernels):
    conv = keras.layers.Conv3D(kernels, kernel_size=(3, 5, 5),
                               strides=(1, 2, 2), activation='relu',
                               padding='same')(inputs)
    pooling = keras.layers.MaxPool3D(pool_size=(1, 2, 2), padding='same', strides=(1, 1, 1))(conv)
    bn = keras.layers.BatchNormalization()(pooling)
    return bn


def _spatial_feature_extraction(inputs, kernels=(32, 32, 64)):
    if len(kernels) <= 0:
        return inputs
    conv = _conv_block(inputs, kernels[0])
    for filters in kernels[1:]:
        conv = _conv_block(conv, filters)
    return conv


def stfm_model(time_slice=12, relevance_distance=5):
    """
    论文：基于时空特征的移动网络流量预测模型
    论文地址：http://kns.cnki.net/kcms/detail/50.1075.TP.20190816.1423.010.html
    三维卷积 + TCN 串联的方式
    :param time_slice: 序列的时间片个，就是序列的长度
    :param relevance_distance: 相关性距离
    :return:
    """
    inputs = _get_inputs(time_slice=time_slice, relevance_distance=relevance_distance)
    spatial_feature = _spatial_feature_extraction(inputs, kernels=(32, 32, 64))
    spatial_feature = keras.layers.Reshape((-1, 1))(spatial_feature)

    time_feature = TCN(nb_filters=24, dilations=(1, 2, 4, 8), activation='relu')(spatial_feature)
    fc = keras.layers.Flatten()(time_feature)

    outputs = keras.layers.Dense(1, activation='relu', use_bias=True)(fc)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def tcn2d_conv_block(input_layers, activation, use_batch_norm, use_layer_norm, dropout_rate):
    added_output = keras.layers.add(input_layers)
    if use_batch_norm:
        added_output = keras.layers.BatchNormalization()(added_output)
    elif use_layer_norm:
        added_output = keras.layers.LayerNormalization()(added_output)

    added_output = keras.layers.Activation(activation)(added_output)
    added_output = keras.layers.Dropout(dropout_rate)(added_output)
    return added_output


# 残差模块
def residual_module(input_layers,
                    filters=8,
                    kernel_size=3,
                    residual_length=2,  # 残差跳跃的卷积数量
                    activation='relu',
                    kernel_initializer='he_normal',
                    dropout_rate=0.3,
                    use_batch_norm=True,
                    use_layer_norm=False):
    """残差模块"""
    conv = input_layers
    for i in range(residual_length):
        conv = keras.layers.Conv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   activation=activation,
                                   padding='same',
                                   kernel_initializer=kernel_initializer
                                   )(conv)
        if use_batch_norm:
            conv = keras.layers.BatchNormalization()(conv)
        elif use_layer_norm:
            conv = keras.layers.LayerNormalization()(conv)

        conv = keras.layers.Activation('relu')(conv)
        conv = keras.layers.Dropout(dropout_rate)(conv)

    # 残差的附加卷积不用激活
    residual = keras.layers.Conv2D(filters=filters,
                                   kernel_size=1,
                                   padding='same',
                                   kernel_initializer=kernel_initializer)(input_layers)
    added_output = keras.layers.add([conv, residual])
    added_output = keras.layers.Activation(activation)(added_output)
    return added_output


# 2dTCN
def tcn2d_model(time_slice=16,
                relevance_distance=7,
                filters=8,
                span=2,
                kernel_size=3,
                layers_count=4,
                activation='relu',
                dropout_rate=0.3,
                use_batch_norm=True,
                use_layer_norm=False):
    """ 二维时间卷积神经网络
    :param relevance_distance: 相关性距离
    :param time_slice: time_slice 长度应该是 kernel_size的 layers_count 次方
    :param filters: 滤波器的个数
    :param span: 跨度
    :param kernel_size: 卷积的尺寸
    :param layers_count: 层数
    :param activation: 激活函数
    :param dropout_rate: dropout rate
    :param use_batch_norm: 批归一化
    :param use_layer_norm: 层归一化
    :return:
    """
    assert time_slice == pow(span, layers_count), '序列的长度必须等于 kernel_size^layers_count'

    inputs_ = _get_inputs(time_slice=time_slice, relevance_distance=relevance_distance)
    # inputs = keras.layers.BatchNormalization()(inputs_)
    inputs = inputs_
    convs = []
    for i in range(0, pow(span, layers_count), span):
        addeds = []
        for k in range(span):
            residual = residual_module(inputs[:, i + k],
                                       filters=filters,
                                       kernel_size=kernel_size,
                                       residual_length=2,
                                       activation=activation,
                                       dropout_rate=dropout_rate,
                                       use_batch_norm=use_batch_norm,
                                       use_layer_norm=use_layer_norm)
            addeds.append(residual)
        added_output = keras.layers.add(addeds)
        convs.append(added_output)

    for i in range(1, layers_count):
        convs_temp = []
        for j in range(0, len(convs), span):
            addeds = []
            for k in range(span):
                residual = residual_module(convs[j + k],
                                           filters=(i + 1) * filters,
                                           kernel_size=kernel_size,
                                           residual_length=2,
                                           activation=activation,
                                           dropout_rate=dropout_rate,
                                           use_batch_norm=use_batch_norm,
                                           use_layer_norm=use_layer_norm)
                addeds.append(residual)
            added_output = keras.layers.add(addeds)
            convs_temp.append(added_output)
        convs = convs_temp

    x = convs[0]
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(2048, activation=activation)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Dense(1024, activation=activation)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Dense(512, activation=activation)(x)
    x = keras.layers.Dense(1)(x)
    outputs = keras.layers.Activation('linear')(x)
    model = keras.models.Model(inputs=inputs_, outputs=outputs)
    return model


def conv3d_model(time_slice=16,
                 relevance_distance=7,
                 filters=8,
                 residual_length=2,  # 残差跳跃的卷积数量
                 kernel_size=3,
                 layers_count=4,
                 activation='relu',
                 dropout_rate=0.3,
                 use_batch_norm=True,
                 use_layer_norm=False):
    # assert time_slice == pow(span, layers_count), '序列的长度必须等于 kernel_size^layers_count'
    # 卷积中使用批归一化，全连接中使用dropout
    inputs_ = _get_inputs(time_slice=time_slice, relevance_distance=relevance_distance)
    # inputs = keras.layers.BatchNormalization()(inputs_)
    inputs = inputs_
    # 3DConv module
    conv3d = inputs
    for i in range(0, layers_count, residual_length):
        conv3d_f = keras.layers.Convolution3D((i + 1) * filters,
                                              # kernel_size=(3, 3, 3),
                                              kernel_size=kernel_size,
                                              activation=activation,
                                              padding='same')(conv3d)
        if use_batch_norm:
            conv3d_f = keras.layers.BatchNormalization()(conv3d_f)
        elif use_layer_norm:
            conv3d_f = keras.layers.LayerNormalization()(conv3d_f)
        conv3d_f = keras.layers.Dropout(dropout_rate)(conv3d_f)
        conv3d_f = keras.layers.Convolution3D((i + 1) * filters,
                                              # kernel_size=(3, 3, 3),
                                              kernel_size=kernel_size,
                                              activation=activation,
                                              padding='same')(conv3d_f)
        if use_batch_norm:
            conv3d_f = keras.layers.BatchNormalization()(conv3d_f)
        elif use_layer_norm:
            conv3d_f = keras.layers.LayerNormalization()(conv3d_f)
        conv3d_f = keras.layers.Dropout(dropout_rate)(conv3d_f)
        # 残差模块的附加卷积
        residual = keras.layers.Convolution3D((i + 1) * filters,
                                              # kernel_size=(3, 3, 3),
                                              kernel_size=1,
                                              activation=None,
                                              padding='same')(conv3d)
        conv3d = keras.layers.add([conv3d_f, residual])
        conv3d = keras.layers.Activation(activation)(conv3d)
        conv3d = keras.layers.MaxPool3D(pool_size=(2, 2, 2))(conv3d)

    x = keras.layers.Flatten()(conv3d)

    x = keras.layers.Dense(4096, activation=activation)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Dense(2048, activation=activation)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Dense(512, activation=activation)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Dense(1)(x)
    outputs = keras.layers.Activation('linear')(x)
    model = keras.models.Model(inputs=inputs_, outputs=outputs)
    return model


# 2DTCN并联了一个3DConv
def tcn2d_and_conv3d_model(time_slice=16,
                           relevance_distance=7,
                           filters=8,
                           span=2,
                           residual_length=2,  # 残差跳跃的卷积数量
                           tcn2d_kernel_size=3,
                           conv3d_kernel_size=3,
                           tcn2d_layers_count=4,
                           conv3d_layers_count=4,  # 3DCNN层数
                           activation='relu',
                           dropout_rate=0.3,
                           use_batch_norm=True,
                           use_layer_norm=False):
    assert time_slice == pow(span, tcn2d_layers_count), '序列的长度必须等于 kernel_size^layers_count'

    inputs_ = _get_inputs(time_slice=time_slice, relevance_distance=relevance_distance)
    # inputs = keras.layers.BatchNormalization()(inputs_)
    inputs = inputs_

    convs = []
    for i in range(0, pow(span, tcn2d_layers_count), span):
        addeds = []
        for k in range(span):
            residual = residual_module(inputs[:, i + k],
                                       filters=filters,
                                       kernel_size=tcn2d_kernel_size,
                                       residual_length=residual_length,
                                       activation=activation,
                                       dropout_rate=dropout_rate,
                                       use_batch_norm=use_batch_norm,
                                       use_layer_norm=use_layer_norm)
            addeds.append(residual)
        added_output = keras.layers.add(addeds)
        convs.append(added_output)

    for i in range(1, tcn2d_layers_count):
        convs_temp = []
        for j in range(0, len(convs), span):
            addeds = []
            for k in range(span):
                residual = residual_module(convs[j + k],
                                           filters=(i + 1) * filters,
                                           kernel_size=tcn2d_kernel_size,
                                           residual_length=residual_length,
                                           activation=activation,
                                           dropout_rate=dropout_rate,
                                           use_batch_norm=use_batch_norm,
                                           use_layer_norm=use_layer_norm)
                addeds.append(residual)
            added_output = keras.layers.add(addeds)
            convs_temp.append(added_output)
        convs = convs_temp
    tcn2d = convs[0]

    # 3DConv module
    conv3d = inputs
    for i in range(0, conv3d_layers_count, residual_length):
        conv3d_f = keras.layers.Convolution3D((i + 1) * filters,
                                              # kernel_size=(3, 3, 3),
                                              kernel_size=conv3d_kernel_size,
                                              activation=activation,
                                              kernel_initializer='he_normal',
                                              padding='same')(conv3d)
        if use_batch_norm:
            conv3d_f = keras.layers.BatchNormalization()(conv3d_f)
        elif use_layer_norm:
            conv3d_f = keras.layers.LayerNormalization()(conv3d_f)
        conv3d_f = keras.layers.Dropout(dropout_rate)(conv3d_f)
        conv3d_f = keras.layers.Convolution3D((i + 2) * filters,
                                              # kernel_size=(3, 3, 3),
                                              kernel_size=conv3d_kernel_size,
                                              activation=activation,
                                              kernel_initializer='he_normal',
                                              padding='same')(conv3d_f)
        if use_batch_norm:
            conv3d_f = keras.layers.BatchNormalization()(conv3d_f)
        elif use_layer_norm:
            conv3d_f = keras.layers.LayerNormalization()(conv3d_f)
        conv3d_f = keras.layers.Dropout(dropout_rate)(conv3d_f)
        # 残差模块的附加卷积
        residual = keras.layers.Convolution3D((i + 2) * filters,
                                              # kernel_size=(3, 3, 3),
                                              kernel_size=1,
                                              activation=None,
                                              kernel_initializer='he_normal',
                                              padding='same')(conv3d)
        conv3d = keras.layers.add([conv3d_f, residual])
        conv3d = keras.layers.Activation(activation)(conv3d)
        # conv3d = keras.layers.MaxPooling3D(pool_size=2)(conv3d)
        # conv3d = maxpooling3d.MaxPooling3D(data_format = 'channels_last')(conv3d)

    conv3d_slice = keras.layers.Lambda(lambda tensor: [tensor[:, i] for i in range(tensor.get_shape()[1])])(conv3d)
    conv3d_slice.append(tcn2d)
    x = keras.layers.add(conv3d_slice)
    # conv3d = keras.layers.Flatten()(conv3d)
    # tcn2d = keras.layers.Flatten()(tcn2d)
    # x = keras.layers.concatenate([conv3d, tcn2d])

    x = keras.layers.Activation(activation)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(2048, activation=activation)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Dense(1024, activation=activation)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Dense(512, activation=activation)(x)
    x = keras.layers.Dense(1)(x)
    outputs = keras.layers.Activation('linear')(x)
    model = keras.models.Model(inputs=inputs_, outputs=outputs)
    return model


# 先经过3DConv在串联一个2DTCN
def conv3d_tcn2d_model(time_slice=16,
                       relevance_distance=7,
                       filters=8,
                       span=2,
                       kernel_size=3,
                       layers_count=4,
                       activation='relu',
                       dropout_rate=0.3,
                       use_batch_norm=True,
                       use_layer_norm=False):
    """先经过3DConv在串联一个2DTCN"""
    assert time_slice == pow(span, layers_count), '序列的长度必须等于 kernel_size^layers_count'
    # 卷积中使用批归一化，全连接中使用dropout
    inputs_ = _get_inputs(time_slice=time_slice, relevance_distance=relevance_distance)
    inputs = keras.layers.BatchNormalization()(inputs_)

    # 3DConv module
    conv3d = inputs
    conv3d = keras.layers.Convolution3D(32,
                                        # kernel_size=(3, 3, 3),
                                        kernel_size=(3, 5, 5),
                                        strides=(1, 2, 2),
                                        activation=activation,
                                        padding='same')(conv3d)
    conv3d = keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 1, 1))(conv3d)
    conv3d = keras.layers.BatchNormalization()(conv3d)

    conv3d = keras.layers.Convolution3D(64,
                                        # kernel_size=(3, 3, 3),
                                        kernel_size=(3, 5, 5),
                                        strides=(1, 2, 2),
                                        activation=activation,
                                        padding='same')(conv3d)
    conv3d = keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 1, 1))(conv3d)
    conv3d = keras.layers.BatchNormalization()(conv3d)

    conv3d = keras.layers.Convolution3D(128,
                                        # kernel_size=(3, 3, 3),
                                        kernel_size=(3, 5, 5),
                                        strides=(1, 2, 2),
                                        activation=activation,
                                        padding='same')(conv3d)
    conv3d = keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 1, 1))(conv3d)
    conv3d = keras.layers.BatchNormalization()(conv3d)

    convs = []
    for i in range(0, pow(span, layers_count), span):
        addeds = []
        for k in range(span):
            conv = keras.layers.Conv2D(filters=filters,
                                       kernel_size=kernel_size,
                                       activation=activation,
                                       padding='same')(conv3d[:, i + k])
            conv = keras.layers.Conv2D(filters=filters,
                                       kernel_size=kernel_size,
                                       activation=activation,
                                       padding='same')(conv)
            addeds.append(conv)
        added = tcn2d_conv_block(addeds, activation, use_batch_norm, use_layer_norm, dropout_rate)
        convs.append(added)

    for i in range(1, layers_count):
        convs_temp = []
        for j in range(0, len(convs), span):
            addeds = []
            for k in range(span):
                conv = keras.layers.Conv2D(filters=(i + 1) * filters,
                                           kernel_size=kernel_size,
                                           activation=activation,
                                           padding='same')(convs[j + k])
                conv = keras.layers.Conv2D(filters=(i + 1) * filters,
                                           kernel_size=kernel_size,
                                           activation=activation,
                                           padding='same')(conv)
                addeds.append(conv)
            added = tcn2d_conv_block(addeds, activation, use_batch_norm, use_layer_norm, dropout_rate)
            convs_temp.append(added)
        convs = convs_temp

    x = convs[0]
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(4096, activation=activation)(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(2048, activation=activation)(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(512, activation=activation)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.models.Model(inputs=inputs_, outputs=outputs)
    return model


if __name__ == '__main__':
    seq_length = 16
    d = 7
    # stn = stn_model(time_slice=12, relevance_distance=5)
    # stn.summary()
    # keras.utils.plot_model(stn, "./model_png/stn.png", show_shapes=True)
    # seq_length = 6
    # d = 7
    # stfm = stfm_model(time_slice=seq_length, relevance_distance=d)
    # stfm.summary()
    # keras.utils.plot_model(stfm, "./model_png/stfm.png", show_shapes=True)
    # stfm = stfm(time_slice=seq_length, relevance_distance=d)
    # stfm.summary()
    # keras.utils.plot_model(stfm, "./model_png/stfm_separableConv2D.png", show_shapes=True)

    # tcn_temp = tcn2d_model(time_slice=seq_length,
    #                        relevance_distance=d,
    #                        dropout_rate=0.5,
    #                        use_batch_norm=False,
    #                        use_layer_norm=True)
    # tcn_temp.summary()
    # keras.utils.plot_model(tcn_temp, "./model_png/tcn2d_model.png", show_shapes=True)

    # tcn_3dconv = tcn2d_conv3d_model()
    # tcn_3dconv.summary()
    # keras.utils.plot_model(tcn_3dconv, "./model_png/tcn_3dconv_1.png", show_shapes=True)

    # conv3d = conv3d_model()
    # conv3d.summary()
    # keras.utils.plot_model(conv3d, "./model_png/conv3d_model.png", show_shapes=True)

    # tcn2d_and_conv3d_model
    tcn2d_and_conv3d = tcn2d_and_conv3d_model(time_slice=seq_length,
                                              relevance_distance=d,
                                              dropout_rate=0.5,
                                              use_batch_norm=False,
                                              use_layer_norm=True)
    tcn2d_and_conv3d.summary()
    keras.utils.plot_model(tcn2d_and_conv3d, "./model_png/tcn2d_and_conv3d_model.png", show_shapes=True)
# Total params: 17,495,393
# Trainable params: 17,489,249
# Non-trainable params: 6,144
