#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Site    : 
# @File    : my2_data_processor.py
# @IDE: PyCharm Community Edition
"""
实现训练数据提供类
"""
import os.path as ops

import cv2
import numpy as np

try:
    from cv2 import cv2
except ImportError:
    pass

VGG_MEAN = [103.939, 116.779, 123.68]    
    

class DataSet(object):
    """
    实现数据集类
    """

    def __init__(self, dataset_info_file):
        """

        :param dataset_info_file:
        """
        self._gt_img_list,   self._gt_depth_list  ,  self._gt_label_list, self._gt_cam_list = self._init_dataset(dataset_info_file)
        self._random_dataset()
        self._next_batch_loop_count = 0

    def _init_dataset(self, dataset_info_file):
        """

        :param dataset_info_file:
        :return:
        """
  
        gt_img_list = []
        gt_gt_list = []
        gt_depth_list = []
        gt_cam_list = []
        assert ops.exists(dataset_info_file), '{:s}　not exsit'.format(dataset_info_file)

        with open(dataset_info_file, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()
                # assert ops.exists(info_tmp[0]) and ops.exists(info_tmp[1])
                                
                path = info_tmp[0]
                img_path = '/***/***/' + path    #You need change the path 
                gt_path = img_path.replace('/image/','/groundtruth_depth/').replace('_sync_image_', '_sync_groundtruth_depth_')
                depth_path = img_path.replace('/image/','/velodyne_raw/').replace('_sync_image_', '_sync_velodyne_raw_')                
                cam_path = img_path.replace('/image/','/intrinsics/').replace('.png','.txt')
                gt_img_list.append(img_path)
                gt_depth_list.append(depth_path)
                gt_gt_list.append(gt_path)
                gt_cam_list.append(cam_path)
                
                

        return gt_img_list, gt_depth_list , gt_gt_list , gt_cam_list

    def _random_dataset(self):
        """

        :return:
        """
        assert len(self._gt_img_list) == len(self._gt_label_list)

        random_idx = np.random.permutation(len(self._gt_img_list))
        new_gt_img_list = []
        new_gt_depth_list = []
        new_gt_label_list = []
        new_gt_cam_list = []
        for index in random_idx:
            new_gt_img_list.append(self._gt_img_list[index])
            new_gt_label_list.append(self._gt_label_list[index])
            new_gt_depth_list.append(self._gt_depth_list[index])
            new_gt_cam_list.append(self._gt_cam_list[index])
        self._gt_img_list = new_gt_img_list
        self._gt_label_list = new_gt_label_list
        self._gt_depth_list = new_gt_depth_list
        self._gt_cam_list = new_gt_cam_list

    def next_batch(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        assert len(self._gt_label_list) == len(self._gt_img_list)

        idx_start = batch_size * self._next_batch_loop_count
        idx_end = batch_size * self._next_batch_loop_count + batch_size

        if idx_end > len(self._gt_label_list):
            self._random_dataset()
            self._next_batch_loop_count = 0
            return self.next_batch(batch_size)
        else:
            gt_img_list = self._gt_img_list[idx_start:idx_end]
            gt_label_list = self._gt_label_list[idx_start:idx_end]
            gt_depth_list = self._gt_depth_list[idx_start:idx_end]
            gt_cam_list = self._gt_cam_list[idx_start:idx_end]
            gt_imgs = []
            gt_labels = []
            gt_depths = []
            gt_cams = []
            gt_masks = []
            xx_ = []
            yy_ = []
            
            for gt_img_path in gt_img_list:
                gt_imgs.append(cv2.imread(gt_img_path, cv2.IMREAD_COLOR) - VGG_MEAN)

            for gt_label_path in gt_label_list:
                label_img = cv2.imread(gt_label_path, -1)
                mask = label_img > 0
                #label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
                depth = label_img.astype(np.float)/255 
                #depth = np.log(depth + 1)
                #depth[label_img == 0] = -1.
                depth = depth[..., np.newaxis]
                gt_labels.append(depth)
                mask = np.array(mask, dtype = np.uint8)
                mask = mask[..., np.newaxis]
                gt_masks.append(mask)
                
                
            for gt_depth_path in gt_depth_list:
                depth_img = cv2.imread(gt_depth_path, -1)
                #label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
                depth = label_img.astype(np.float)/255 
                #depth = np.log(label_img.astype(np.float) + 1)
                #depth[label_img == 0] = -1.
                depth = depth[..., np.newaxis]
                gt_depths.append(depth)
                
            x = np.arange(0, 1216)
            y = np.arange(0, 352)
            xv, yv = np.meshgrid(x, y)
            xv = xv[..., np.newaxis]
            yv = yv[..., np.newaxis]
            for gt_cam_path in gt_cam_list:
                f = open(gt_cam_path)
                sp = f.readline().split(' ')            
                cx = sp[2]
                cy = sp[5]
                fx = sp[0]
                fy = sp[4]
                cam = np.array([cx,cy,fx,fy], dtype = np.float32)
                gt_cams.append(cam)
                xx_.append(xv)
                yy_.append(yv)
            self._next_batch_loop_count += 1
            return gt_imgs, gt_depths, gt_labels , gt_masks, gt_cams, xx_, yy_


if __name__ == '__main__':
    val = DataSet('/***/***/train_img.txt')
    a1, a2, a3, a4 = val.next_batch(10)
    b1, b2, b3, b4 = val.next_batch(10)
    c1, c2, c3, c4 = val.next_batch(10)
    dd, d2, d3, d4 = val.next_batch(10)
    print (a4[0].shape)
    print (np.max(a4[0]), '   ' , np.min(a4[0]))
    print (a1[0].shape)
    #print (a2)
    
