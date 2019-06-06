#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Site    : 
# @File    : resnet_encoder.py
# @IDE: PyCharm Community Edition
"""
实现一个基于VGG16的特征编码类
"""
from collections import OrderedDict

import tensorflow as tf
import sys
sys.path.append('/home2/***/dense_depth/lanenet-lane-detection/') # You need to change the path
from encoder_decoder_model import cnn_basenet
from encoder_decoder_model import resnet_layer

class RESNETEncoder(cnn_basenet.CNNBaseModel):
    """
    实现了一个基于vgg16的特征编码类
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(RESNETEncoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def _conv_stage(self, input_tensor, k_size, out_dims, name,
                    stride=1, pad='SAME'):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(inputdata=input_tensor, out_channel=out_dims,
                               kernel_size=k_size, stride=stride,
                               use_bias=False, padding=pad, name='conv')

            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def _conv_stage_no(self, input_tensor, k_size, out_dims, name,
                    stride=1, pad='SAME'):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(inputdata=input_tensor, out_channel=out_dims,
                               kernel_size=k_size, stride=stride,
                               use_bias=False, padding=pad, name='conv')


            relu = self.relu(inputdata=conv, name='relu')

        return relu
        
    def _fc_stage(self, input_tensor, out_dims, name, use_bias=False):
        """

        :param input_tensor:
        :param out_dims:
        :param name:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name):
            fc = self.fullyconnect(inputdata=input_tensor, out_dim=out_dims, use_bias=use_bias,
                                   name='fc')

            bn = self.layerbn(inputdata=fc, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def encode(self, input_tensor1, input_tensor2, name):
        """
        根据vgg16框架对输入的tensor进行编码
        :param input_tensor:
        :param name:
        :param flags:
        :return: 输出vgg16编码特征
        """
        ret = OrderedDict()
        
        with tf.variable_scope(name):
            conv_img = self._conv_stage(input_tensor=input_tensor1, k_size=3,
                                        out_dims=48, name='input_tensor1')    
            conv_SP = self._conv_stage_no(input_tensor=input_tensor2, k_size=3,
                                        out_dims=16, name='input_tensor2')    
                                                  
            input_ = tf.concat([conv_img, conv_SP], -1)
            block_fn = 'basic_block' 
            repetitions = [3, 4, 6, 3]
            ret = resnet_layer.ResnetBuilder2.build(input_, block_fn, repetitions)

        return ret

if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.float32, shape=[1, 2048, 2048, 3], name='input1')
    b = tf.placeholder(dtype=tf.float32, shape=[1, 2048, 2048, 3], name='input2')
    encoder = RESNETEncoder(phase=tf.constant('train', dtype=tf.string))
    ret = encoder.encode(a,b, name='encode')
    for layer_name, layer_info in ret.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))
