#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
测试LaneNet模型
"""
import os
import os.path as ops
import argparse
import time
import math

import tensorflow as tf
import glob
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from lanenet_model import mylanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess
from config import global_config
import matplotlib as mpl
import scipy.io as scio
import copy
CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=32)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()
    
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)


def test_lanenet(image_path, weights_path, use_gpu):
    """

    :param image_path:
    :param weights_path:
    :param use_gpu:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    log.info('开始读取图像数据并进行预处理')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imwrite('ori.png', image)
    image_vis = image
    
    sparse = cv2.imread(image_path.replace('/image/','/velodyne_raw/').replace('_sync_image_', '_sync_velodyne_raw_') ,-1)
    print('sparse_max: ', np.max(sparse))
    max_depth = np.max(sparse)
    image = np.array(image, dtype = np.float32)
    sparse = np.array(sparse, dtype = np.float32)
    #sparse = np.log(sparse + 1)
    sparse = sparse/255.
    sparse = sparse[..., np.newaxis]
    #image = cv2.resize(image, (1216, 352), interpolation=cv2.INTER_LINEAR)
    image = image - VGG_MEAN
    log.info('图像读取完毕, 耗时: {:.5f}s'.format(time.time() - t_start))

    input_tensor1 = tf.placeholder(dtype=tf.float32, shape=[1, 352, 1216, 3], name='input_tensor1')
    input_tensor2 = tf.placeholder(dtype=tf.float32, shape=[1, 352, 1216, 1], name='input_tensor2')
    phase_tensor = tf.constant('train', tf.string)

    net = mylanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='resnet')
    pred = net.inference(input_tensor1=input_tensor1, input_tensor2=input_tensor2, name='dense_loss')

    #cluster = lanenet_cluster.LaneNetCluster()
    #postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        image = image[np.newaxis,...]
        sparse = sparse[np.newaxis, ...]
        print (image.shape)
        print (sparse.shape)
        pred_result = sess.run([pred],
                                                        feed_dict={input_tensor1: image,
                                                        input_tensor2: sparse
                                                        })
                  
        t_cost = time.time() - t_start
        log.info('单张图像车道线预测耗时: {:.5f}s'.format(t_cost))
        print (np.max(pred_result[0]))
        print (np.min(pred_result[0]))
        print (pred_result[0].shape)
        pred_result = pred_result[0][0]
        pred_result[pred_result < 0] = 0
        
        
        
        #pred_result = np.exp(pred_result) - 1
        
        #pred_result[pred_result > 15000 ] = 0
        
        cv2.imwrite('test.png', pred_result/np.max(pred_result)*255)
        #pred_result[pred_result >2] = 1/ pred_result[pred_result >5]
        print (pred_result.shape)
        print (np.max(pred_result))
        print (np.min(pred_result))
        pred_result0 =  np.array(pred_result, dtype = np.int)
        
        pred_result1 =  np.array(pred_result*255, dtype = np.int)
        
        min_ = np.min(pred_result)
        max_ = np.max(pred_result)
        
        a = pred_result / max_ *255 
        pred_result2 =  np.array(a, dtype = np.uint8)
        
        im_color = cv2.applyColorMap(pred_result2, cv2.COLORMAP_JET)
        
        cv2.imwrite('testC.png', im_color)
        
        
        data = {};
            
        data['A'] = pred_result1
        dataNew = '11.mat'
        scio.savemat(dataNew, {'A':data['A']})

        #cv2.imwrite('test.png', np.log(pred_result+1)/np.max(np.log(pred_result+1))*255)
        #binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])
        #mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
        #                                   instance_seg_ret=instance_seg_image[0])
        # mask_image = cluster.get_lane_mask_v2(instance_seg_ret=instance_seg_image[0])
        # mask_image = cv2.resize(mask_image, (image_vis.shape[1], image_vis.shape[0]),
        #                         interpolation=cv2.INTER_LINEAR)

        #ele_mex = np.max(instance_seg_image[0], axis=(0, 1))
        #for i in range(3):
        #    if ele_mex[i] == 0:
        #        scale = 1
        #    else:
        #        scale = 255 / ele_mex[i]
        #    instance_seg_image[0][:, :, i] *= int(scale)
        #embedding_image = np.array(instance_seg_image[0], np.uint8)
        # cv2.imwrite('embedding_mask.png', embedding_image)

        # mask_image = cluster.get_lane_mask_v2(instance_seg_ret=embedding_image)
        # mask_image = cv2.resize(mask_image, (image_vis.shape[1], image_vis.shape[0]),
        #                         interpolation=cv2.INTER_LINEAR)

        #cv2.imwrite('binary_ret.png', binary_seg_image[0] * 255)
        #cv2.imwrite('instance_ret.png', embedding_image)

        #plt.figure('mask_image')
        #plt.imshow(pred_result[0])
        #norm = colors.BoundaryNorm(np.arange(0,100),256)
        #a1 = copy.copy(pred_result[:,:,0])
        #a1 = np.array(a1, dtype = np.float32)
        #a1[a1>1] = 65536/a1[a1>1]
        #print(np.max(a1))
        plt.imsave('colormap.jpg',pred_result1[:,:,0], cmap='plasma')
        #plt.imsave('colormap2.jpg',a1, cmap='plasma')

        


        #plt.figure('src_image')
        #plt.imshow(image_vis[:, :, (2, 1, 0)])
        #plt.figure('instance_image')
        #plt.imshow(embedding_image[:, :, (2, 1, 0)])
        #plt.figure('binary_image')
        #plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        #plt.show()

    sess.close()

    return


def test_lanenet_batch(image_dir, weights_path, batch_size, use_gpu, save_dir=None):
    """

    :param image_dir:
    :param weights_path:
    :param batch_size:
    :param use_gpu:
    :param save_dir:
    :return:
    """
    assert ops.exists(image_dir), '{:s} not exist'.format(image_dir)

    log.info('开始获取图像文件路径...')
    image_path_list = glob.glob('{:s}/*.jpg'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/*.png'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/*.jpeg'.format(image_dir), recursive=True)

    input_tensor1 = tf.placeholder(dtype=tf.float32, shape=[1, 352, 1216, 3], name='input_tensor1')
    input_tensor2 = tf.placeholder(dtype=tf.float32, shape=[1, 352, 1216, 1], name='input_tensor2')
    
    phase_tensor = tf.constant('train', tf.string)

    net = mylanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='resnet')
    pred = net.inference(input_tensor1=input_tensor1, input_tensor2=input_tensor2, name='dense_loss')

    #cluster = lanenet_cluster.LaneNetCluster()
    #postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        epoch_nums = int(math.ceil(len(image_path_list) / batch_size))
        
        for epoch in range(epoch_nums):
            log.info('[Epoch:{:d}] 开始图像读取和预处理...'.format(epoch))
            t_start = time.time()
            image_path_epoch = image_path_list[epoch * batch_size:(epoch + 1) * batch_size]
            image_list_epoch = [cv2.imread(tmp, cv2.IMREAD_COLOR) for tmp in image_path_epoch]
            image_vis_list = image_list_epoch
            image_list_epoch = [tmp - VGG_MEAN for tmp in image_list_epoch]
            sparse_list = []
            for j in range(len(image_path_epoch)):
                print (image_path_epoch[j])
                print (image_path_epoch[j].replace('/test_image/','/test_velodyne/').replace('_sync_image_', '_sync_velodyne_raw_'))
                sparse = cv2.imread(image_path_epoch[j].replace('/test_image/','/test_velodyne/').replace('_sync_image_', '_sync_velodyne_raw_') ,-1)
                
                sparse = np.array(sparse, dtype = np.float32)/255
                sparse = sparse[..., np.newaxis]
                print (sparse.shape)
                sparse_list.append(sparse)
            
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] 预处理{:d}张图像, 共耗时: {:.5f}s, 平均每张耗时: {:.5f}'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            t_start = time.time()
            #binary_seg_images, instance_seg_images = sess.run(
            #    [binary_seg_ret, instance_seg_ret], feed_dict={input_tensor: image_list_epoch})
                
            pred_result = sess.run([pred],
                                                        feed_dict={input_tensor1: image_list_epoch,
                                                        input_tensor2: sparse_list
                                                        })            
                
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] 预测{:d}张图像车道线, 共耗时: {:.5f}s, 平均每张耗时: {:.5f}s'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))
            
            for j in range(len(image_path_epoch)):
                pred_result_each = pred_result[j][0]
                pred_result_each[pred_result_each < 0] = 0 
                
                image_name = ops.split(image_path_epoch[j])[1]
                image_save_path = ops.join(save_dir, image_name)    

                #max_ = np.max(pred_result_each)                
                #a = pred_result_each / max_ *255 
                #pred_result2 =  np.array(a, dtype = np.uint8)
                
                #im_color = cv2.applyColorMap(pred_result2, cv2.COLORMAP_JET)
                
                #cv2.imwrite(image_save_path, im_color)
                
                
                pred_result_each = np.array(pred_result_each*255, dtype = np.uint16)

                cv2.imwrite(image_save_path, pred_result_each)
                
                
                
            
            '''
            cluster_time = []
            for index, binary_seg_image in enumerate(binary_seg_images):
                t_start = time.time()
                binary_seg_image = postprocessor.postprocess(binary_seg_image)
                mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image,
                                                   instance_seg_ret=instance_seg_images[index])
                cluster_time.append(time.time() - t_start)
                mask_image = cv2.resize(mask_image, (image_vis_list[index].shape[1],
                                                     image_vis_list[index].shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

                if save_dir is None:
                    plt.ion()
                    plt.figure('mask_image')
                    plt.imshow(mask_image[:, :, (2, 1, 0)])
                    plt.figure('src_image')
                    plt.imshow(image_vis_list[index][:, :, (2, 1, 0)])
                    plt.pause(3.0)
                    plt.show()
                    plt.ioff()

                if save_dir is not None:
                    mask_image = cv2.addWeighted(image_vis_list[index], 1.0, mask_image, 1.0, 0)
                    image_name = ops.split(image_path_epoch[index])[1]
                    image_save_path = ops.join(save_dir, image_name)
                    cv2.imwrite(image_save_path, mask_image)
                    # log.info('[Epoch:{:d}] Detection image {:s} complete'.format(epoch, image_name))
            log.info('[Epoch:{:d}] 进行{:d}张图像车道线聚类, 共耗时: {:.5f}s, 平均每张耗时: {:.5f}'.format(
                epoch, len(image_path_epoch), np.sum(cluster_time), np.mean(cluster_time)))
            '''
    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if args.save_dir is not None and not ops.exists(args.save_dir):
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)

    if args.is_batch.lower() == 'false':
        # test hnet model on single image
        test_lanenet(args.image_path, args.weights_path, args.use_gpu)
    else:
        # test hnet model on a batch of image
        test_lanenet_batch(image_dir=args.image_path, weights_path=args.weights_path,
                           save_dir=args.save_dir, use_gpu=args.use_gpu, batch_size=args.batch_size)

                           
                           