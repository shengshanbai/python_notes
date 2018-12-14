# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:06:11 2018

@author: admin
"""
import tensorflow as tf
import mnist_util

BATCH_SIZE=128

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

class LeNet:
    
    def __init__(self,dataset):
        self.dataset=dataset
        
    def _create_input(self):
        with tf.name_scope('input'):
            self.input=tf.placeholder(tf.float32,[-1,mnist_util.IMAGE_SIZE,mnist_util.IMAGE_SIZE,1],name='input')
            
    def _create_conv1(self):
        with tf.variable_scope('layer-conv1'):
            conv1_weights=tf.get_variable("weight",
                                          [CONV1_SIZE,CONV1_SIZE,1,CONV1_DEEP],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases=tf.get_variable("bias",[CONV1_DEEP],
                                         initializer=tf.constant_initializer(0.0)))
            # 使用边长为5，深度为32的过滤器，过滤器移动的步长为1，且使用全0填充
            conv1 = tf.nn.conv2d(self.input, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))