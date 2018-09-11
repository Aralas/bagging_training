
# coding: utf-8

# In[1]:


import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import numpy as np
import random
import time
from multiprocessing import Pool, TimeoutError


# In[2]:

## define tool functions

def weight_variable(shape, initialization):
    if initialization == 'xavier':
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.Variable(initializer(shape))
    elif initialization == 'random':
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

"""
卷积和池化，使用卷积步长为1（stride size）,0边距（padding size）
池化用简单传统的2x2大小的模板做max pooling
"""

def conv2d_bn(x, W, b, activation='relu', stride=1):
    h_conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding = 'SAME') + b
    if activation == 'relu':
        h_conv = tf.nn.relu(h_conv)
    elif activation == 'softmax':
        h_conv == tf.nn.softmax(h_conv)
    return h_conv
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    # x(input)  : [batch, in_height, in_width, in_channels]
    # W(filter) : [filter_height, filter_width, in_channels, out_channels]
    # strides   : The stride of the sliding window for each dimension of input.
    #             For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1]

def max_pool_2x2(x, size=2, stride=2):
    return tf.nn.max_pool(x, ksize = [1, size, size, 1],
                          strides = [1, stride, stride, 1], padding = 'SAME')
    # tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
    # x(value)              : [batch, height, width, channels]
    # ksize(pool大小)        : A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
    # strides(pool滑动大小)   : A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.  

def fully_bn_layer(x, W, b, activation='relu'):
    h_fc = tf.matmul(x, W) + b
    if activation == 'relu':
        h_fc = tf.nn.relu(h_fc)
    elif activation == 'softmax':
        h_fc = tf.nn.softmax(h_fc)
    elif activation == 'sigmoid':
        h_fc = tf.nn.sigmoid(h_fc)
    return h_fc


# In[3]:

def generate_model(train_X, train_Y, label, n):
    tf.reset_default_graph()
    keep_prob = tf.placeholder("float")
    
    """
    第一层 卷积层
    x_image(batch, 28, 28, 1) -> h_pool1(batch, 14, 14, 8)
    """
    x = tf.placeholder(tf.float32,[None, 784])    
    x_image = tf.reshape(x, [-1, 28, 28, 1]) #最后一维代表通道数目，如果是rgb则为3 
    W_conv1 = weight_variable([5, 5, 1, 8], initialization)
    b_conv1 = bias_variable([8])
    h_conv1 = conv2d_bn(x_image, W_conv1, b_conv1, 'relu') 
    # x_image -> [batch, in_height, in_width, in_channels]
    #            [batch, 28, 28, 1]
    # W_conv1 -> [filter_height, filter_width, in_channels, out_channels]
    #            [5, 5, 1, 32]
    # output  -> [batch, out_height, out_width, out_channels]
    #            [batch, 28, 28, 32]
    h_pool1 = max_pool_2x2(h_conv1)
    # h_conv1 -> [batch, in_height, in_weight, in_channels]
    #            [batch, 28, 28, 32]
    # output  -> [batch, out_height, out_weight, out_channels]
    #            [batch, 14, 14, 32]

    """
    第二层 卷积层
    h_pool1(batch, 14, 14, 8) -> h_pool2(batch, 7, 7, 8)
    """
    W_conv2 = weight_variable([5, 5, 8, 8], initialization)
    b_conv2 = bias_variable([8])
    h_conv2 = conv2d_bn(h_pool1, W_conv2, b_conv2, 'relu')
    # h_pool1 -> [batch, 14, 14, 32]
    # W_conv2 -> [5, 5, 32, 32]
    # output  -> [batch, 14, 14, 32]
    h_pool2 = max_pool_2x2(h_conv2)
    # h_conv2 -> [batch, 14, 14, 32]
    # output  -> [batch, 7, 7, 32]

    """
    第三层 全连接层

    h_pool2(batch, 7, 7, 8) -> h_fc1(1, 256)
    """
    W_fc1 = weight_variable([7 * 7 * 8, 256], initialization)
    b_fc1 = bias_variable([256])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 8])
    h_fc1 = fully_bn_layer(h_pool2_flat, W_fc1, b_fc1, 'relu')
    h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

    """
    第四层 Softmax输出层
    """
    W_fc2 = weight_variable([256, 1], initialization)
    b_fc2 = bias_variable([1])
    y_conv = fully_bn_layer(h_fc1, W_fc2, b_fc2, activation='sigmoid')

    """
    训练和评估模型

    ADAM优化器来做梯度最速下降,feed_dict中加入参数keep_prob控制dropout比例
    """
    y_ = tf.placeholder("float", [None, 1])
    loss = tf.reduce_mean(tf.square(y_conv - y_))  
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss) #使用adam优化器来以0.0001的学习率来进行微调
        
    
    with tf.Session() as sess: 
        init = tf.global_variables_initializer()  
        sess.run(init)
        func = lambda x: 1 if x>=0.5 else 0 
        for i in range(max_interations): 
            indeces_small = random.sample(range(len(train_X)), batch_size)
            sess.run(train_step, feed_dict = {x:train_X[indeces_small], y_:train_Y[indeces_small], keep_prob:dropout_probability}) 
            if i % 10 == 0:
                train_y = sess.run(y_conv, feed_dict = {x:train_X, y_:train_Y, keep_prob:1})                               
                y_hat = [[func(train_y[i][0])] for i in range(len(train_y))]
                acc = 1 - np.mean(np.square(y_hat-train_Y))                
                print('label %d, n_subset %d, step %d, accuracy %g' %(label, n, i, acc))  
                if acc>0.93:
                    break
            
        return(sess.run(W_conv1), sess.run(b_conv1), sess.run(W_conv2), sess.run(b_conv2),
           sess.run(W_fc1), sess.run(b_fc1), sess.run(W_fc2), sess.run(b_fc2))
    


# In[4]:

def bagging_binary_classifier(label, N_subset):
    W_conv1 = np.zeros([5, 5, 1, 8])
    b_conv1 = np.zeros([8])
    W_conv2 = np.zeros([5, 5, 8, 8])
    b_conv2 = np.zeros([8])
    W_fc1 = np.zeros([7 * 7 * 8, 256])
    b_fc1 = np.zeros([256])
    W_fc2 = np.zeros((256, 1))
    b_fc2 = np.zeros([1])
    
    for n in range(N_subset):
        index_train = positive_indeces[label*training_size:(label+1)*training_size] +                         random.sample(unlabeled_indeces, training_size)
        X = mnist.train.images[index_train]
        Y = np.array([1]*training_size+[0]*training_size).reshape(2*training_size,1)
        (W_conv1_, b_conv1_, W_conv2_, b_conv2_, W_fc1_, b_fc1_, W_fc2_, b_fc2_) = generate_model(X, Y, label, n)
        W_conv1 = W_conv1 + W_conv1_ 
        b_conv1 = b_conv1 + b_conv1_
        W_conv2 = W_conv2 + W_conv2_
        b_conv2 = b_conv2 + b_conv2_
        W_fc1 = W_fc1 + W_fc1_
        b_fc1 = b_fc1 + b_fc1_
        W_fc2 = W_fc2 + W_fc2_
        b_fc2 = b_fc2 + b_fc2_
    
    return(W_conv1/N_subset, b_conv1/N_subset, W_conv2/N_subset, b_conv2/N_subset, 
           W_fc1/N_subset, b_fc1/N_subset, W_fc2/N_subset, b_fc2/N_subset)


# In[5]:

## define hyper parameters
file_index = 1
random_seed = 10
training_size = 100
learning_rate = 0.001
dropout_probability = 0.5
initialization = 'random'
max_interations = 200
batch_size = 128

## download data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Y_train = mnist.train.labels
Y_train_label = np.where(Y_train==1)[1]

random.seed(random_seed)
positive_indeces = []     # randomly choose positive training set
for label in range(10):
    indeces = list(np.where(Y_train_label==label)[0])
    index_slice = random.sample(indeces, training_size)
    positive_indeces = positive_indeces + index_slice
train_data = mnist.train.images[positive_indeces]
train_label = mnist.train.labels[positive_indeces]  
unlabeled_indeces = set(range(len(Y_train)))-set(positive_indeces)


# In[8]:

## define hyper parameters
file_index = 1
random_seed = 10
training_size = 100
learning_rate = 0.001
dropout_probability = 0.5
initialization = 'random'
max_interations = 200
batch_size = 128

## download data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Y_train = mnist.train.labels
Y_train_label = np.where(Y_train==1)[1]

random.seed(random_seed)
positive_indeces = []     # randomly choose positive training set
for label in range(10):
    indeces = list(np.where(Y_train_label==label)[0])
    index_slice = random.sample(indeces, training_size)
    positive_indeces = positive_indeces + index_slice
train_data = mnist.train.images[positive_indeces]
train_label = mnist.train.labels[positive_indeces]  
unlabeled_indeces = set(range(len(Y_train)))-set(positive_indeces)
label = 1
N_subset = 10
filename = './bagging_weights/'+initialization+'_datasize'+str(training_size)+'_bagging'+str(N_subset)+'_seed'+str(random_seed)+'/label'+str(label)
(W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2) = bagging_binary_classifier(label, N_subset)
np.savez(filename, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)


# In[9]:

## define hyper parameters
file_index = 1
random_seed = 10
training_size = 100
learning_rate = 0.001
dropout_probability = 0.5
initialization = 'random'
max_interations = 200
batch_size = 128

## download data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Y_train = mnist.train.labels
Y_train_label = np.where(Y_train==1)[1]

random.seed(random_seed)
positive_indeces = []     # randomly choose positive training set
for label in range(10):
    indeces = list(np.where(Y_train_label==label)[0])
    index_slice = random.sample(indeces, training_size)
    positive_indeces = positive_indeces + index_slice
train_data = mnist.train.images[positive_indeces]
train_label = mnist.train.labels[positive_indeces]  
unlabeled_indeces = set(range(len(Y_train)))-set(positive_indeces)
label = 2
N_subset = 10
filename = './bagging_weights/'+initialization+'_datasize'+str(training_size)+'_bagging'+str(N_subset)+'_seed'+str(random_seed)+'/label'+str(label)
(W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2) = bagging_binary_classifier(label, N_subset)
np.savez(filename, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)


# In[10]:

## define hyper parameters
file_index = 1
random_seed = 10
training_size = 100
learning_rate = 0.001
dropout_probability = 0.5
initialization = 'random'
max_interations = 200
batch_size = 128

## download data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Y_train = mnist.train.labels
Y_train_label = np.where(Y_train==1)[1]

random.seed(random_seed)
positive_indeces = []     # randomly choose positive training set
for label in range(10):
    indeces = list(np.where(Y_train_label==label)[0])
    index_slice = random.sample(indeces, training_size)
    positive_indeces = positive_indeces + index_slice
train_data = mnist.train.images[positive_indeces]
train_label = mnist.train.labels[positive_indeces]  
unlabeled_indeces = set(range(len(Y_train)))-set(positive_indeces)
label = 3
N_subset = 10
filename = './bagging_weights/'+initialization+'_datasize'+str(training_size)+'_bagging'+str(N_subset)+'_seed'+str(random_seed)+'/label'+str(label)
(W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2) = bagging_binary_classifier(label, N_subset)
np.savez(filename, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)


# In[11]:

## define hyper parameters
file_index = 1
random_seed = 10
training_size = 100
learning_rate = 0.001
dropout_probability = 0.5
initialization = 'random'
max_interations = 200
batch_size = 128

## download data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Y_train = mnist.train.labels
Y_train_label = np.where(Y_train==1)[1]

random.seed(random_seed)
positive_indeces = []     # randomly choose positive training set
for label in range(10):
    indeces = list(np.where(Y_train_label==label)[0])
    index_slice = random.sample(indeces, training_size)
    positive_indeces = positive_indeces + index_slice
train_data = mnist.train.images[positive_indeces]
train_label = mnist.train.labels[positive_indeces]  
unlabeled_indeces = set(range(len(Y_train)))-set(positive_indeces)
label = 4
N_subset = 10
filename = './bagging_weights/'+initialization+'_datasize'+str(training_size)+'_bagging'+str(N_subset)+'_seed'+str(random_seed)+'/label'+str(label)
(W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2) = bagging_binary_classifier(label, N_subset)
np.savez(filename, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)


# In[12]:

## define hyper parameters
file_index = 1
random_seed = 10
training_size = 100
learning_rate = 0.001
dropout_probability = 0.5
initialization = 'random'
max_interations = 200
batch_size = 128

## download data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Y_train = mnist.train.labels
Y_train_label = np.where(Y_train==1)[1]

random.seed(random_seed)
positive_indeces = []     # randomly choose positive training set
for label in range(10):
    indeces = list(np.where(Y_train_label==label)[0])
    index_slice = random.sample(indeces, training_size)
    positive_indeces = positive_indeces + index_slice
train_data = mnist.train.images[positive_indeces]
train_label = mnist.train.labels[positive_indeces]  
unlabeled_indeces = set(range(len(Y_train)))-set(positive_indeces)
label = 5
N_subset = 10
filename = './bagging_weights/'+initialization+'_datasize'+str(training_size)+'_bagging'+str(N_subset)+'_seed'+str(random_seed)+'/label'+str(label)
(W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2) = bagging_binary_classifier(label, N_subset)
np.savez(filename, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)


# In[13]:

## define hyper parameters
file_index = 1
random_seed = 10
training_size = 100
learning_rate = 0.001
dropout_probability = 0.5
initialization = 'random'
max_interations = 200
batch_size = 128

## download data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Y_train = mnist.train.labels
Y_train_label = np.where(Y_train==1)[1]

random.seed(random_seed)
positive_indeces = []     # randomly choose positive training set
for label in range(10):
    indeces = list(np.where(Y_train_label==label)[0])
    index_slice = random.sample(indeces, training_size)
    positive_indeces = positive_indeces + index_slice
train_data = mnist.train.images[positive_indeces]
train_label = mnist.train.labels[positive_indeces]  
unlabeled_indeces = set(range(len(Y_train)))-set(positive_indeces)
label = 6
N_subset = 10
filename = './bagging_weights/'+initialization+'_datasize'+str(training_size)+'_bagging'+str(N_subset)+'_seed'+str(random_seed)+'/label'+str(label)
(W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2) = bagging_binary_classifier(label, N_subset)
np.savez(filename, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)


# In[14]:

## define hyper parameters
file_index = 1
random_seed = 10
training_size = 100
learning_rate = 0.001
dropout_probability = 0.5
initialization = 'random'
max_interations = 200
batch_size = 128

## download data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Y_train = mnist.train.labels
Y_train_label = np.where(Y_train==1)[1]

random.seed(random_seed)
positive_indeces = []     # randomly choose positive training set
for label in range(10):
    indeces = list(np.where(Y_train_label==label)[0])
    index_slice = random.sample(indeces, training_size)
    positive_indeces = positive_indeces + index_slice
train_data = mnist.train.images[positive_indeces]
train_label = mnist.train.labels[positive_indeces]  
unlabeled_indeces = set(range(len(Y_train)))-set(positive_indeces)
label = 7
N_subset = 10
filename = './bagging_weights/'+initialization+'_datasize'+str(training_size)+'_bagging'+str(N_subset)+'_seed'+str(random_seed)+'/label'+str(label)
(W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2) = bagging_binary_classifier(label, N_subset)
np.savez(filename, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)


# In[15]:

## define hyper parameters
file_index = 1
random_seed = 10
training_size = 100
learning_rate = 0.001
dropout_probability = 0.5
initialization = 'random'
max_interations = 200
batch_size = 128

## download data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Y_train = mnist.train.labels
Y_train_label = np.where(Y_train==1)[1]

random.seed(random_seed)
positive_indeces = []     # randomly choose positive training set
for label in range(10):
    indeces = list(np.where(Y_train_label==label)[0])
    index_slice = random.sample(indeces, training_size)
    positive_indeces = positive_indeces + index_slice
train_data = mnist.train.images[positive_indeces]
train_label = mnist.train.labels[positive_indeces]  
unlabeled_indeces = set(range(len(Y_train)))-set(positive_indeces)
label = 8
N_subset = 10
filename = './bagging_weights/'+initialization+'_datasize'+str(training_size)+'_bagging'+str(N_subset)+'_seed'+str(random_seed)+'/label'+str(label)
(W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2) = bagging_binary_classifier(label, N_subset)
np.savez(filename, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)


# In[16]:

## define hyper parameters
file_index = 1
random_seed = 10
training_size = 100
learning_rate = 0.001
dropout_probability = 0.5
initialization = 'random'
max_interations = 200
batch_size = 128

## download data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Y_train = mnist.train.labels
Y_train_label = np.where(Y_train==1)[1]

random.seed(random_seed)
positive_indeces = []     # randomly choose positive training set
for label in range(10):
    indeces = list(np.where(Y_train_label==label)[0])
    index_slice = random.sample(indeces, training_size)
    positive_indeces = positive_indeces + index_slice
train_data = mnist.train.images[positive_indeces]
train_label = mnist.train.labels[positive_indeces]  
unlabeled_indeces = set(range(len(Y_train)))-set(positive_indeces)
label = 9
N_subset = 10
filename = './bagging_weights/'+initialization+'_datasize'+str(training_size)+'_bagging'+str(N_subset)+'_seed'+str(random_seed)+'/label'+str(label)
(W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2) = bagging_binary_classifier(label, N_subset)
np.savez(filename, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)


# In[17]:

## define hyper parameters
file_index = 1
random_seed = 30
training_size = 100
learning_rate = 0.001
dropout_probability = 0.5
initialization = 'random'
max_interations = 200
batch_size = 128

## download data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Y_train = mnist.train.labels
Y_train_label = np.where(Y_train==1)[1]

random.seed(random_seed)
positive_indeces = []     # randomly choose positive training set
for label in range(10):
    indeces = list(np.where(Y_train_label==label)[0])
    index_slice = random.sample(indeces, training_size)
    positive_indeces = positive_indeces + index_slice
train_data = mnist.train.images[positive_indeces]
train_label = mnist.train.labels[positive_indeces]  
unlabeled_indeces = set(range(len(Y_train)))-set(positive_indeces)

N_subset = 10
for label in range(10):
    filename = './bagging_weights/'+initialization+'_datasize'+str(training_size)+'_bagging'+str(N_subset)+'_seed'+str(random_seed)+'/label'+str(label)
    (W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2) = bagging_binary_classifier(label, N_subset)
    np.savez(filename, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)


# In[18]:

## define hyper parameters
file_index = 1
random_seed = 50
training_size = 100
learning_rate = 0.001
dropout_probability = 0.5
initialization = 'random'
max_interations = 200
batch_size = 128

## download data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Y_train = mnist.train.labels
Y_train_label = np.where(Y_train==1)[1]

random.seed(random_seed)
positive_indeces = []     # randomly choose positive training set
for label in range(10):
    indeces = list(np.where(Y_train_label==label)[0])
    index_slice = random.sample(indeces, training_size)
    positive_indeces = positive_indeces + index_slice
train_data = mnist.train.images[positive_indeces]
train_label = mnist.train.labels[positive_indeces]  
unlabeled_indeces = set(range(len(Y_train)))-set(positive_indeces)

N_subset = 10
for label in range(10):
    filename = './bagging_weights/'+initialization+'_datasize'+str(training_size)+'_bagging'+str(N_subset)+'_seed'+str(random_seed)+'/label'+str(label)
    (W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2) = bagging_binary_classifier(label, N_subset)
    np.savez(filename, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)


# In[ ]:



