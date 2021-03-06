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
batch_size = 64
g_lr = 0.0001
d_lr = 0.0001
beta1 = 0.0
beta2 = 0.9
lambd = 1 / 4

# Attention Mechanism
def attention(x, channels, scope):

    with tf.variable_scope(scope, reuse= tf.AUTO_REUSE):
        
        # f, g, and h Weight Matrices -- 1x1 Convolutions
        f = conv(x, channels // 8, kernel_size= 1, strides= 1, scope= 'f_conv')
        g = conv(x, channels // 8, kernel_size= 1, strides= 1, scope= 'g_conv')
        h = conv(x, channels, kernel_size= 1, strides= 1, scope= 'h_conv')
        
        s = tf.matmul(flatten(g), flatten(f), transpose_b= True)
        beta = tf.nn.softmax(s, axis= -1)
        
        x_shape = (x).get_shape().as_list()
        o = tf.matmul(beta, flatten(h))
        o = tf.reshape(o, x_shape)
        
        gamma = tf.get_variable('gamma', [1], initializer= tf.constant_initializer(0.0)) 
        out = tf.add((gamma * o), x)

        return out

# Convolutional Generator
def generator(z, is_training= True):
    
    with tf.variable_scope('g', reuse= tf.AUTO_REUSE):
        
        dl1 = tf.layers.dense(z, 8192, activation= tf.nn.leaky_relu)
        dl1 = tf.reshape(dl1, [-1, 4, 4, 512])
        
        deconv1 = deconv(dl1, 256, scope= 'deconv1')
        deconv1 = batch_normalization(deconv1, is_training= is_training)
        deconv1 = tf.nn.relu(deconv1)

        deconv2 = deconv(deconv1, 128, scope= 'deconv2')
        deconv2 = batch_normalization(deconv2, is_training= is_training)
        deconv2 = tf.nn.relu(deconv2)
        
        deconv3 = deconv(deconv2, 64, scope= 'deconv3')
        deconv3 = batch_normalization(deconv3, is_training= is_training)
        deconv3 = tf.nn.relu(deconv3)
        
        att_deconv3 = attention(deconv3, 64, 'deconv3_att')
        
        deconv4 = deconv(att_deconv3, 32, scope= 'deconv4')
        deconv4 = batch_normalization(deconv4, is_training= is_training)
        deconv4 = tf.nn.relu(deconv4)

        deconv5 = deconv(deconv4, 3, scope= 'deconv5')
        deconv5 = tf.nn.tanh(deconv5)
            
        return deconv5

# Convolutional Discriminator
def discriminator(x, is_training= True):
    
    with tf.variable_scope('d', reuse= tf.AUTO_REUSE):
        
        conv1 = conv(x, 32, scope= 'conv1')
        conv1 = batch_normalization(conv1, is_training= is_training)
        conv1 = tf.nn.leaky_relu(conv1)
        
        conv2 = conv(conv1, 64, scope= 'conv2')
        conv2 = batch_normalization(conv2, is_training= is_training)
        conv2 = tf.nn.leaky_relu(conv2)
        
        att_conv2 = attention(conv2, 64, 'conv2_att')
        
        conv3 = conv(att_conv2, 128, scope= 'conv3')
        conv3 = batch_normalization(conv3, is_training= is_training)
        conv3 = tf.nn.leaky_relu(conv3)

        conv4 = conv(conv3, 256, scope= 'conv4')
        conv4 = batch_normalization(conv4, is_training= is_training)
        conv4 = tf.nn.leaky_relu(conv4)
        
        conv5 = conv(conv4, 512, scope= 'conv5')
        conv5 = batch_normalization(conv5, is_training= is_training)
        conv5 = tf.nn.leaky_relu(conv5)

        flatt_conv5 = tf.layers.flatten(conv5)
        dl1 = dense(flatt_conv5, 1)
        
        return dl1
            
# Placeholders
x = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
x_pert = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
z = tf.placeholder(tf.float32, [batch_size, 128])
it = tf.placeholder(tf.bool)

dr = discriminator(x, is_training= it)
gf = generator(z, is_training= it)
df = discriminator(gf, is_training= it)

# Loss
r_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= dr, labels= tf.ones_like(dr)))
f_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= df, labels= tf.zeros_like(df)))
d_loss = tf.add(r_loss, f_loss)

alpha = tf.random_uniform(shape= (x).get_shape().as_list(), minval= 0.0, maxval= 1.0)
x_diff = x_pert - x
interpolate = x + (alpha * x_diff)
disc_out = discriminator(interpolate, is_training= True)
grads = tf.gradients(disc_out, [interpolate])[0]
slope = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices= [1]))
gradient_penalty = tf.reduce_mean((slope - 1.0) ** 2)
d_loss = d_loss + (lambd * gradient_penalty)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= df, labels= tf.ones_like(df)))

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd' in var.name]
g_vars = [var for var in t_vars if 'g' in var.name]

# Optimization
d_opt = tf.train.AdamOptimizer(d_lr, beta1= beta1, beta2= beta2).minimize(d_loss, var_list= d_vars)
g_opt = tf.train.AdamOptimizer(g_lr, beta1= beta1, beta2= beta2).minimize(g_loss, var_list= g_vars)
        
sess.run(tf.global_variables_initializer())

batches_in_epoch = 20000 // batch_size # Where 20000 == Number of training images
images = []

# Checkpoints
saver = tf.train.Saver()
save_dir = '/Users/Masterchief7269/Desktop/Programming Header/Python/Models/SAGAN/'

# Training
for epoch_iter in range (epochs):
    for batch_iter in range (1, batches_in_epoch):
        b_x = ret_data(batch_size, batch_iter)
        b_z = np.random.uniform(-1.0, 1.0, size= [batch_size, 128])
        x_p = ret_pert(b_x)
        sess.run(d_opt, feed_dict= {x: b_x, x_pert: x_p, z: b_z, it: True})
        sess.run(g_opt, feed_dict= {x: b_x, x_pert: x_p, z: b_z, it: True})
        
        if batch_iter % 15 == 0:
            gl = sess.run(g_loss, feed_dict= {z: b_z, it: False})
            dl = sess.run(d_loss, feed_dict= {x: b_x, x_pert: x_p, z: b_z, it: False})
            
            print('Epoch %s, Batch %s / %s, D-loss: %s, G-loss: %s' %(epoch_iter, batch_iter, batches_in_epoch, dl, gl))
            
        if batch_iter % 25 == 0:
            g_img = sess.run(gf, feed_dict= {x: b_x, x_pert: x_p, z: b_z, it: False})
            images.append(g_img)

        if batch_iter % 100 == 0:
            saver.save(sess, save_dir)
