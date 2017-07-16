import tensorflow as tf
import numpy as np

def Concat(tensors, axis, name='concat'):
    # tensors: list of tensors
    # axis: concat axis
    
    return tf.concat(tensors, axis, name)

def ReLu(x, name):
    # x: input tensor (Tensor)
    # name: variable scope (str)
    l = tf.nn.relu(x)
    return l

def PReLu(x, alpha, name):
    x = ((1 + alpha) * x + (1 - alpha) * tf.abs(x))
    ret = tf.multiply(x, 0.5, name=name)
    return ret

def LeakyReLu(x, alpha, name):
    return tf.maximum(x, alpha * x, name=name)

def Conv2D(x, filter_shape, out_dim, strides, padding, name, initializer=tf.contrib.layers.variance_scaling_initializer(), reuse=False):
    # x: input tensor (float32)[n, w, h, c]
    # filter_shape: conv2d filter (int)[w, h]
    # out_dim: output channels (int)
    # strides: conv2d stride size (int)
    # padding: padding type (str)
    # name: variable scope (str)
           
    with tf.variable_scope(name, reuse=reuse) as scope:
        in_dim = x.get_shape()[-1]
        w = tf.get_variable('w', shape=filter_shape + [in_dim, out_dim], initializer=initializer)
        b = tf.get_variable('b', shape=[out_dim], initializer=tf.constant_initializer(0.0))
        l = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding, name='conv2d')
        l = tf.nn.bias_add(l, b, name='bias_add')
    return l

def FC(x, out_dim, name, initializer=tf.contrib.layers.variance_scaling_initializer(), reuse=False):
    # x: input tensor (float32)[n, in_dim]
    # out_dim: output channels (int)
    # name: variable scope (str)

    x = tf.contrib.layers.flatten(x)
    with tf.variable_scope(name, reuse=reuse) as scope:
        in_dim = x.get_shape()[-1]
        w = tf.get_variable('w', shape=[in_dim, out_dim], initializer=initializer)
        b = tf.get_variable('b', shape=[out_dim], initializer=tf.constant_initializer(0.0))
        l = tf.add(tf.matmul(x, w), b, name='add')
    return l

def Deconv2D(x, filter_shape, output_shape, out_dim, strides, padding, name, reuse=False):
    # x: input tensor (float32) [n, w, h, c]
    # filter_shape: conv2d filter (int)[w, h]
    # out_dim: output channels (int)
    # strides: conv2d stride size (int)
    # padding: padding type (str)
    # name: variable scope (str)

    with tf.variable_scope(name, reuse=reuse) as scope:
        in_dim = x.get_shape()[-1]
        w = tf.get_variable('w', shape=filter_shape + [out_dim, in_dim], initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
        b = tf.get_variable('b', shape=[out_dim], initializer=tf.constant_initializer(0.0))
        l = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, strides, strides, 1], padding=padding, name='deconv2d')
        l = tf.nn.bias_add(l, b, name='bias_add')
    return l

def HuberLoss(x, delta=1.0, name='loss'):
    sqrcost = tf.square(x)
    abscost = tf.abs(x)
    return tf.where(abscost < delta,
                    sqrcost * 0.5,
                    abscost * delta - 0.5 * delta ** 2)

