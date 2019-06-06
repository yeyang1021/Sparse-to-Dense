#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-1-29 下午2:38
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : dilation_decoder.py
# @IDE: PyCharm Community Edition
"""
实现一个全卷积网络解码类
"""
import tensorflow as tf

from encoder_decoder_model import cnn_basenet
from encoder_decoder_model import vgg_encoder
from encoder_decoder_model import dense_encoder
from encoder_decoder_model import resnet_encoder


class FCNDecoder(cnn_basenet.CNNBaseModel):
    """
    实现一个全卷积解码类
    """
    def __init__(self):
        """

        """
        super(FCNDecoder, self).__init__()

    def decode(self, input_tensor_dict, decode_layer_list, name):
        """
        解码特征信息反卷积还原
        :param input_tensor_dict:
        :param decode_layer_list: 需要解码的层名称需要由深到浅顺序写
                                  eg. ['pool5', 'pool4', 'pool3']
        :param name:
        :return:
        """
        ret = dict()

        with tf.variable_scope(name):
            # score stage 1
            input_tensor = input_tensor_dict[decode_layer_list[0]]['data']

            #score = self.deconv2d(inputdata=input_tensor, out_channel=256,
            #                    kernel_size=4, stride=2, use_bias=False, name='score_origin')
            score = self.conv2d(inputdata=input_tensor, out_channel=256,
                                kernel_size=1, use_bias=False, name='score_origin')
                                
                                
            decode_layer_list = decode_layer_list[1:]
            print (decode_layer_list)
            for i in range(len(decode_layer_list)):
                deconv = self.deconv2d(inputdata=score, out_channel=64, kernel_size=4,
                                       stride=2, use_bias=False, name='deconv_{:d}'.format(i + 1))
                input_tensor = input_tensor_dict[decode_layer_list[i]]['data']
                score = self.conv2d(inputdata=input_tensor, out_channel=64,
                                    kernel_size=1, use_bias=False, name='score_{:d}'.format(i + 1))
                fused = tf.add(deconv, score, name='fuse_{:d}'.format(i + 1))
                score = fused

            #deconv_final = self.deconv2d(inputdata=score, out_channel=64, kernel_size=16,
            #                            stride=8, use_bias=False, name='deconv_final')

            score_final = self.conv2d(inputdata=score, out_channel=1,
                                      kernel_size=1, use_bias=False, name='score_final')

            ret['map'] = score_final
            #ret['deconv'] = deconv_final

        #     # score stage 1
        #     input_tensor = input_tensor_dict['pool5']
        #
        #     score_1 = self.conv2d(inputdata=input_tensor, out_channel=2,
        #                           kernel_size=1, use_bias=False, name='score_1')
        #
        #     # decode stage 1
        #     deconv_1 = self.deconv2d(inputdata=score_1, out_channel=2, kernel_size=4,
        #                              stride=2, use_bias=False, name='deconv_1')
        #
        #     # score stage 2
        #     score_2 = self.conv2d(inputdata=input_tensor_dict['pool4'], out_channel=2,
        #                           kernel_size=1, use_bias=False, name='score_2')
        #
        #     # fuse stage 1
        #     fuse_1 = tf.add(deconv_1, score_2, name='fuse_1')
        #
        #     # decode stage 2
        #     deconv_2 = self.deconv2d(inputdata=fuse_1, out_channel=2, kernel_size=4,
        #                              stride=2, use_bias=False, name='deconv_2')
        #
        #     # score stage 3
        #     score_3 = self.conv2d(inputdata=input_tensor_dict['pool3'], out_channel=2,
        #                           kernel_size=1, use_bias=False, name='score_3')
        #
        #     # fuse stage 2
        #     fuse_2 = tf.add(deconv_2, score_3, name='fuse_2')
        #
        #     # decode stage 3
        #     deconv_3 = self.deconv2d(inputdata=fuse_2, out_channel=2, kernel_size=16,
        #                              stride=8, use_bias=False, name='deconv_3')
        #
        #     # score stage 4
        #     score_4 = self.conv2d(inputdata=deconv_3, out_channel=2,
        #                           kernel_size=1, use_bias=False, name='score_4')
        #
        # ret['logits'] = score_4

        return ret


if __name__ == '__main__':

    resnet_encoder = resnet_encoder.RESNETEncoder(phase=tf.constant('train', tf.string))
    #dense_encoder = dense_encoder.DenseEncoder(l=40, growthrate=12,
    #                                           with_bc=True, phase='train', n=5)
    decoder = FCNDecoder()

    in_tensor1 = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 3],
                               name='input')
    in_tensor2 = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 6],
                               name='input')
    resnet_encoder_encode_ret = resnet_encoder.encode(in_tensor1, in_tensor2, name='resnet_encoder')
    #dense_encode_ret = dense_encoder.encode(in_tensor, name='dense_encoder')
    decode_ret = decoder.decode(resnet_encoder_encode_ret, name='decoder',
                                decode_layer_list=['block_03_03',
                                                   'block_02_06',
                                                   'block_01_04',
                                                   'block_00_03',
                                                   'block_before'])
    #for layer_name, layer_info in decode_ret.items():
    #    print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))
    print (decode_ret.items())                                               
                                                   
                                                   
                                                   
