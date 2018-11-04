# Chandler Supple, 11/4/2018

# Libraries
from ops import *
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# Session
sess = tf.InteractiveSession()

# Parameters, Hyperparameters
epochs = 128
batch_size = 32
g_lr = 0.0001
d_lr = 0.0004
beta1 = 0.0
beta2 = 0.9

# Attention Mechanism
def attention(x, channels, scope):

    with tf.variable_scope(scope, reuse= tf.AUTO_REUSE):
        
        # f, g, and h Weight Matrices -- 1x1 Convolutions
        f = conv(x, channels // 8, kernel_size= 1, strides= 1, pad= 0, scope= 'f_conv')
        g = conv(x, channels // 8, kernel_size= 1, strides= 1, pad= 0, scope= 'g_conv')
        h = conv(x, channels, kernel_size= 1, strides= 1, pad= 0, scope= 'h_conv')
        
        s = tf.matmul(flatten(g), flatten(f), transpose_b= True)
        beta = tf.nn.softmax(s, axis= -1)
        
        x_shape = (x).get_shape().as_list()
        o = tf.matmul(beta, flatten(h))
        o = tf.reshape(o, x_shape)
        
        gamma = tf.get_variable('gamma', [1], initializer= tf.constant_initializer(0.0)) 
        out = (gamma * o) + x

        return out

# Convolutional Generator
def generator(z, is_training= True):
    
    with tf.variable_scope('g', reuse= tf.AUTO_REUSE):
        
        dl1 = tf.layers.dense(z, 8192, activation= tf.nn.leaky_relu)
        dl1 = tf.reshape(dl1, [-1, 4, 4, 512])
        
        deconv1 = deconv(dl1, 256, scope= 'deconv1')
        deconv1 = batch_normalization(deconv1, is_training= is_training)
        deconv1 = tf.nn.relu(deconv1)
        att_deconv1 = attention(deconv1, 256, 'deconv1_att')

        deconv2 = deconv(att_deconv1, 128, strides= 1, scope= 'deconv2')
        deconv2 = batch_normalization(deconv2, is_training= is_training)
        deconv2 = tf.nn.relu(deconv2)
        att_deconv2 = attention(deconv2, 128, 'deconv2_att')
        
        deconv3 = deconv(att_deconv2, 64, strides= 1, scope= 'deconv3')
        deconv3 = batch_normalization(deconv3, is_training= is_training)
        deconv3 = tf.nn.relu(deconv3)
        att_deconv3 = attention(deconv3, 64, 'deconv3_att')
        
        deconv4 = deconv(att_deconv3, 32, scope= 'deconv4')
        deconv4 = batch_normalization(deconv4, is_training= is_training)
        deconv4 = tf.nn.relu(deconv4)
        att_deconv4 = attention(deconv4, 32, 'deconv4_att')
        
        deconv5 = deconv(att_deconv4, 3, scope= 'deconv5')
        deconv5 = tf.nn.tanh(deconv5)
            
        return deconv5

# Convolutional Discriminator
def discriminator(x, is_training= True):
    
    with tf.variable_scope('d', reuse= tf.AUTO_REUSE):
        
        conv1 = conv(x, 32, scope= 'conv1')
        conv1 = batch_normalization(conv1, is_training= is_training)
        conv1 = tf.nn.relu(conv1)
        att_conv1 = attention(conv1, 32, 'conv1_att')
        
        conv2 = conv(att_conv1, 64, scope= 'conv2')
        conv2 = batch_normalization(conv2, is_training= is_training)
        conv2 = tf.nn.relu(conv2)
        att_conv2 = attention(conv2, 64, 'conv2_att')
        
        conv3 = conv(att_conv2, 128, scope= 'conv3')
        conv3 = batch_normalization(conv3, is_training= is_training)
        conv3 = tf.nn.relu(conv3)
        att_conv3 = attention(conv3, 128, 'conv3_att')
        
        conv4 = conv(att_conv3, 256, scope= 'conv4')
        conv4 = batch_normalization(conv4, is_training= is_training)
        conv4 = tf.nn.relu(conv4)
        att_conv4 = attention(conv4, 256, 'conv4_att')

        flatt_conv4 = tf.layers.flatten(att_conv4)  
        dl1 = tf.layers.dense(flatt_conv4, 1, activation= tf.nn.sigmoid)
        
        return dl1
            
# Placeholders
x = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
z = tf.placeholder(tf.float32, [batch_size, 100])
it = tf.placeholder(tf.bool)

dr = discriminator(x, is_training= it)
gf = generator(z, is_training= it)
df = discriminator(gf, is_training= it)

# Loss
r_loss = -1 * tf.reduce_mean(dr)
f_loss = tf.reduce_mean(df)
d_loss = r_loss + f_loss

g_loss = -1 * tf.reduce_mean(df)

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd' in var.name]
g_vars = [var for var in t_vars if 'g' in var.name]

# Optimization
d_opt = tf.train.AdamOptimizer(d_lr, beta1= beta1, beta2= beta2).minimize(d_loss, var_list= d_vars)
g_opt = tf.train.AdamOptimizer(g_lr, beta1= beta1, beta2= beta2).minimize(g_loss, var_list= g_vars)
        
sess.run(tf.global_variables_initializer())

# Dataset
batches_in_epoch = 50000 // batch_size
data = ret_data(batch_size, batches_in_epoch)

images = []

# Training
for epoch_iter in range (epochs):
    for batch_iter in range (batches_in_epoch):
        b_x = data[batch_iter]
        b_z = np.random.uniform(-1.0, 1.0, size= [batch_size, 100])
        sess.run(d_opt, feed_dict= {x: b_x, z: b_z, it: True})
        sess.run(g_opt, feed_dict= {x: b_x, z: b_z, it: True})
        
        if batch_iter % 15 == 0:
            gl = sess.run(g_loss, feed_dict= {z: b_z, it: False})
            dl = sess.run(d_loss, feed_dict= {x: b_x, z: b_z, it: False})
            
            print('Epoch %s, Batch %s / %s, D-loss: %s, G-loss: %s' %(epoch_iter, batch_iter, batches_in_epoch, dl, gl))
            
        if batch_iter % 25 == 0:
            g_img = sess.run(gf, feed_dict= {x: b_x, z: b_z, it: False})
            images.append(g_img)
