#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Site    : 
# @File    : mylanenet1_merge_model.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet模型
"""
import tensorflow as tf

from encoder_decoder_model import vgg_encoder
from encoder_decoder_model import fcn_decoder
from encoder_decoder_model import dense_encoder
from encoder_decoder_model import cnn_basenet
from encoder_decoder_model import myfcn_decoder
from encoder_decoder_model import resnet_encoder
from lanenet_model import lanenet_discriminative_loss


class LaneNet(cnn_basenet.CNNBaseModel):
    """
    实现语义分割模型
    """
    def __init__(self, phase, net_flag='resnet'):
        """

        """
        super(LaneNet, self).__init__()
        self._net_flag = net_flag
        self._phase = phase
        if self._net_flag == 'vgg':
            self._encoder = vgg_encoder.VGG16Encoder(phase=phase)
        elif self._net_flag == 'resnet':
            self._encoder = resnet_encoder.RESNETEncoder(phase=phase)
        elif self._net_flag == 'dense':
            self._encoder = dense_encoder.DenseEncoder(l=20, growthrate=8,
                                                       with_bc=True,
                                                       phase=self._phase,
                                                       n=5)
        self._decoder = myfcn_decoder.FCNDecoder()
        return

    def __str__(self):
        """

        :return:
        """
        info = 'Semantic Segmentation use {:s} as basenet to encode'.format(self._net_flag)
        return info

    def _build_model(self, input_tensor1, input_tensor2,name):
        """
        前向传播过程
        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # first encode
            encode_ret = self._encoder.encode(input_tensor1=input_tensor1, input_tensor2=input_tensor2, name='encode')

            # second decode
            if self._net_flag.lower() == 'resnet':
                decode_ret = self._decoder.decode(encode_ret, name='decoder',
                                decode_layer_list=['block_03_03',
                                                   'block_02_06',
                                                   'block_01_04',
                                                   'block_00_03',
                                                   'block_before'])
                return decode_ret


    def compute_loss(self, input_tensor1, input_tensor2, gt_label, mask_label, cam_label, xx_label, yy_label,  name='loss'):
        """
        计算LaneNet模型损失函数
        :param input_tensor:
        :param binary_label:
        :param instance_label:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 前向传播获取logits
            inference_ret = self._build_model(input_tensor1=input_tensor1, input_tensor2=input_tensor2, name='inference')
            # 计算二值分割损失函数
            decode_logits = inference_ret['map']
            
            decode_logits = decode_logits * mask_label
            
            idx = tf.where(mask_label>0)
            select_logits = tf.gather_nd(decode_logits, idx)
            xx_label = tf.divide(cam_label[:,0]- xx_label , cam_label[:,2])
            yy_label = tf.divide(cam_label[:,1]- yy_label , cam_label[:,3])
            select_gt = tf.gather_nd(gt_label, idx)
            select_xx = tf.gather_nd(xx_label, idx)
            select_yy = tf.gather_nd(yy_label, idx)
            
            predx = tf.multiply((select_xx), select_logits ) 
            predy = tf.multiply((select_yy), select_logits )

            gtx = tf.multiply((select_xx), select_gt ) 
            gty = tf.multiply((select_yy), select_gt )            
            
            x_loss = tf.losses.absolute_difference(
                gtx,
                predx,
                weights=1.0,
                scope=None,
                loss_collection=tf.GraphKeys.LOSSES,
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
            )             

            y_loss = tf.losses.absolute_difference(
                gty,
                predy,
                weights=1.0,
                scope=None,
                loss_collection=tf.GraphKeys.LOSSES,
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
            )               
            
            dense_loss = tf.losses.absolute_difference(
                gt_label,
                decode_logits,
                weights=1.0,
                scope=None,
                loss_collection=tf.GraphKeys.LOSSES,
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
            )             
            total_loss = 1.5*dense_loss + 0.1*x_loss + 0.1 * y_loss
            ret = {
                'total_loss': total_loss,
                'dense_loss': dense_loss,
                'x_loss': x_loss,
                'y_loss': y_loss
            }

            return ret

    def inference(self, input_tensor1, input_tensor2,  name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 前向传播获取logits
            inference_ret = self._build_model(input_tensor1=input_tensor1, input_tensor2=input_tensor2, name='inference')
            # 计算二值分割损失函数
            decode_logits = inference_ret['map']
            

            return decode_logits


if __name__ == '__main__':
    model = LaneNet(tf.constant('train', dtype=tf.string))
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    binary_label = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 1], name='label')
    instance_label = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 1], name='label')
    instance_label = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 1], name='label')
    ret = model.compute_loss(input_tensor1=input_tensor, input_tensor2=input_tensor, gt_label=instance_label, mask_label=instance_label,  name='loss')
    print(ret['total_loss'])
