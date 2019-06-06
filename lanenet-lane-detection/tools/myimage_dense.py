#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-18 下午7:31
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : train_lanenet.py
# @IDE: PyCharm Community Edition
"""
训练lanenet模型
"""
import argparse
import math
import os
import os.path as ops
import time

import cv2
import glog as log
import numpy as np
import tensorflow as tf

try:
    from cv2 import cv2
except ImportError:
    pass

from config import global_config
from lanenet_model import mylanenet_merge_model
from lanenet_model import mylanenetImage_merge_model
from data_provider import my1_data_processor

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, help='The training dataset dir path')
    parser.add_argument('--net', type=str, help='Which base net work to use', default='vgg')
    parser.add_argument('--weights_path', type=str, help='The pretrained weights path')

    return parser.parse_args()


def train_net(dataset_dir, weights_path=None, net_flag='vgg'):
    """

    :param dataset_dir:
    :param net_flag: choose which base network to use
    :param weights_path:
    :return:
    """
    train_dataset_file = ops.join(dataset_dir, 'train.txt')
    val_dataset_file = ops.join(dataset_dir, 'val.txt')

    assert ops.exists(train_dataset_file)

    train_dataset = my1_data_processor.DataSet(train_dataset_file)
    val_dataset = my1_data_processor.DataSet(val_dataset_file)

    input_tensor1 = tf.placeholder(dtype=tf.float32,
                                  shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                         CFG.TRAIN.IMG_WIDTH, 3],
                                  name='input_tensor1')
    input_tensor2 = tf.placeholder(dtype=tf.float32,
                                         shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                                CFG.TRAIN.IMG_WIDTH, 1],
                                         name='input_tensor2')
    gt_label = tf.placeholder(dtype=tf.float32,
                                         shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                                CFG.TRAIN.IMG_WIDTH, 1],
                                         name='gt_label')                                         
    mask_label = tf.placeholder(dtype=tf.float32,
                                         shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                                CFG.TRAIN.IMG_WIDTH, 1],
                                         name='mask_label')                                           
                                         
                                           
                                           
                                           
    phase = tf.placeholder(dtype=tf.string, shape=None, name='net_phase')

    net = mylanenetImage_merge_model.LaneNet(net_flag=net_flag, phase=phase)

    # calculate the loss
    compute_ret = net.compute_loss(input_tensor1=input_tensor1, input_tensor2=input_tensor2,
                                   gt_label=gt_label, mask_label=mask_label, name='dense_loss')
    total_loss = compute_ret['total_loss']



    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(CFG.TRAIN.LEARNING_RATE, global_step,
                                               5000, 0.96, staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=
                                           learning_rate).minimize(loss=total_loss,
                                                                   var_list=tf.trainable_variables(),
                                                                   global_step=global_step)

    # Set tf saver
    saver = tf.train.Saver()
    model_save_dir = 'model/culane_lanenet'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'dense_depthnet_{:s}_{:s}.ckpt'.format(net_flag, str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set tf summary
    tboard_save_path = 'tboard/dense_depthnet/{:s}'.format(net_flag)
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    train_cost_scalar = tf.summary.scalar(name='train_cost', tensor=total_loss)
    val_cost_scalar = tf.summary.scalar(name='val_cost', tensor=total_loss)

    learning_rate_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)
    train_merge_summary_op = tf.summary.merge([ train_cost_scalar,
                                               learning_rate_scalar])
    val_merge_summary_op = tf.summary.merge([ val_cost_scalar])

    # Set sess configuration
    sess_config = tf.ConfigProto(device_count={'GPU': 1})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    log.info('Global configuration is as follows:')
    log.info(CFG)

    with sess.as_default():

        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/lanenet_model.pb'.format(model_save_dir))

        if weights_path is None:
            log.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        # 加载预训练参数
        if net_flag == 'vgg' and weights_path is None:
            pretrained_weights = np.load(
                './data/vgg16.npy',
                encoding='latin1').item()

            for vv in tf.trainable_variables():
                weights_key = vv.name.split('/')[-3]
                try:
                    weights = pretrained_weights[weights_key][0]
                    _op = tf.assign(vv, weights)
                    sess.run(_op)
                except Exception as e:
                    continue

        train_cost_time_mean = []
        val_cost_time_mean = []
        for epoch in range(train_epochs):
            # training part
            t_start = time.time()

            gt_imgs, depth_labels, gt_labels, mask_labels = train_dataset.next_batch(CFG.TRAIN.BATCH_SIZE)
            
            phase_train = 'train'

            _, c,  train_summary,   = \
                sess.run([optimizer, total_loss,
                          train_merge_summary_op],
                         feed_dict={input_tensor1: gt_imgs,
                                    input_tensor2: depth_labels,
                                    gt_label: gt_labels,
                                    mask_label: mask_labels,
                                    phase: phase_train})

            if math.isnan(c) :
                log.error('cost is: {:.5f}'.format(c))

                return


            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            summary_writer.add_summary(summary=train_summary, global_step=epoch)

            # validation part
            gt_imgs_val, depth_labels_val, gt_labels_val, mask_labels_val \
                = val_dataset.next_batch(CFG.TRAIN.VAL_BATCH_SIZE)
            
            phase_val = 'test'

            t_start_val = time.time()
            c_val, val_summary = \
                sess.run([total_loss, val_merge_summary_op],
                         feed_dict={input_tensor1: gt_imgs_val,
                                    input_tensor2: depth_labels_val,
                                    gt_label: gt_labels_val,
                                    mask_label: mask_labels_val,
                                    phase: phase_val})


            summary_writer.add_summary(val_summary, global_step=epoch)

            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                log.info('Epoch: {:d} total_loss= {:6f}'
                         ' mean_cost_time= {:5f}s '.
                         format(epoch + 1, c, np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()

            if epoch % CFG.TRAIN.TEST_DISPLAY_STEP == 0:
                log.info('Epoch_Val: {:d} total_loss= {:6f} '
                         'mean_cost_time= {:5f}s '.
                         format(epoch + 1, c_val, np.mean(val_cost_time_mean)))
                val_cost_time_mean.clear()

            if epoch % 2000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # train lanenet
    train_net(args.dataset_dir, args.weights_path, net_flag=args.net)
