# Chandler Supple, 10/4/2018

import tensorflow as tf
from keras.datasets import cifar10
init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

def l2_normalization(x, epsilon= 1e-12):
    return x / (tf.reduce_sum(x ** 2) ** 0.5 + epsilon)
    
def spectral_normalization(x):
    x_shape = (x).get_shape().as_list()
    x = tf.reshape(x, [-1, x_shape[-1]])
    u = tf.get_variable('u', [1, x_shape[-1]], initializer= tf.truncated_normal_initializer(), trainable= False)
    v_und = tf.matmul(u, tf.transpose(x))
    v_hat = l2_normalization(v_und)
    u_und = tf.matmul(v_und, x)
    u_hat = l2_normalization(u_und)        
    sigma = tf.matmul(tf.matmul(v_hat, x), tf.transpose(u_hat))
    x_norm = x / sigma
    
    with tf.control_dependencies([u.assign(u_hat)]):
        resh_x_norm = tf.reshape(x_norm, x_shape)
        
    return resh_x_norm
    
def conv(x, channels, kernel_size= 4, strides= 2, padding= 'SAME', pad= 1, scope= 'conv'):
    
    with tf.variable_scope(scope):
        x_pd = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        w = tf.get_variable('w', shape= [kernel_size, kernel_size, (x_pd).get_shape().as_list()[-1], channels], initializer= init)
        conv_out = tf.nn.conv2d(x_pd, spectral_normalization(w), strides= [1, strides, strides, 1], padding= padding)
        b = tf.get_variable('b', [channels], initializer= tf.constant_initializer(0.0))
        conv_out = tf.nn.bias_add(conv_out, b)
        
        return conv_out
        
def deconv(x, channels, kernel_size= 4, strides= 2, padding= 'SAME', scope= 'deconv'):
    
    with tf.variable_scope(scope):
        x_shape = (x).get_shape().as_list()
        o_shape = [x_shape[0], x_shape[1] * strides, x_shape[2] * strides, channels]
        w = tf.get_variable('w', shape= [kernel_size, kernel_size, channels, (x).get_shape().as_list()[-1]], initializer= init)
        deconv_out = tf.nn.conv2d_transpose(x, spectral_normalization(w), strides= [1, strides, strides, 1], padding= padding, output_shape= o_shape)
        b = tf.get_variable('b', [channels], initializer= tf.constant_initializer(0.0))
        deconv_out = tf.nn.bias_add(deconv_out, b)
        
        return deconv_out
    
def flatten(x):
    x_shape = tf.shape(x)
    return tf.reshape(x, shape= [x_shape[0], -1, x_shape[-1]])
    
def batch_normalization(x, is_training= True):
    return tf.contrib.layers.batch_norm(x, decay= 0.9, epsilon= 1e-05, center= True, scale= True, updates_collections= None, is_training= is_training)
    
def ret_data(batch_size, batches_in_epoch):
    (x_tr, y_tr), _ = cifar10.load_data()
    batches = []
    
    for batch_iter in range (batches_in_epoch):
        batch_x = x_tr[batch_iter * batch_size: (batch_iter * batch_size) + batch_size]
        batches.append(batch_x)
        
    return batches
