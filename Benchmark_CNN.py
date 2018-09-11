
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
        h_fc == tf.nn.softmax(h_fc)
    return h_fc


# In[3]:

def generate_model(test_filename, train_X, train_Y, test_X, test_Y):
    tf.reset_default_graph()
    keep_prob = tf.placeholder("float")
    
    """
    第一层 卷积层
    x_image(batch, 28, 28, 1) -> h_pool1(batch, 14, 14, 32)
    """
    x = tf.placeholder(tf.float32,[None, 784])    
    x_image = tf.reshape(x, [-1, 28, 28, 1]) #最后一维代表通道数目，如果是rgb则为3 
    W_conv1 = weight_variable([5, 5, 1, 32], initialization)
    b_conv1 = bias_variable([32])
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
    h_pool1(batch, 14, 14, 32) -> h_pool2(batch, 7, 7, 32)
    """
    W_conv2 = weight_variable([5, 5, 32, 32], initialization)
    b_conv2 = bias_variable([32])
    h_conv2 = conv2d_bn(h_pool1, W_conv2, b_conv2, 'relu')
    # h_pool1 -> [batch, 14, 14, 32]
    # W_conv2 -> [5, 5, 32, 32]
    # output  -> [batch, 14, 14, 32]
    h_pool2 = max_pool_2x2(h_conv2)
    # h_conv2 -> [batch, 14, 14, 32]
    # output  -> [batch, 7, 7, 32]

    """
    第三层 全连接层

    h_pool2(batch, 7, 7, 32) -> h_fc1(1, 1024)
    """
    W_fc1 = weight_variable([7 * 7 * 32, 1024], initialization)
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 32])
    h_fc1 = fully_bn_layer(h_pool2_flat, W_fc1, b_fc1)
    h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

    """
    第四层 Softmax输出层
    """
    W_fc2 = weight_variable([1024, 10], initialization)
    b_fc2 = bias_variable([10])
    y_conv = fully_bn_layer(h_fc1, W_fc2, b_fc2, activation='None')

    """
    训练和评估模型

    ADAM优化器来做梯度最速下降,feed_dict中加入参数keep_prob控制dropout比例
    """
    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv)) #计算交叉熵   
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy) #使用adam优化器来以0.0001的学习率来进行微调
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) #判断预测标签和实际标签是否匹配
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
        
    test_record = open(test_filename, 'w')

    with tf.Session() as sess: 
        init = tf.global_variables_initializer()  
        sess.run(init)
        for i in range(max_interations): 
            indeces_small = random.sample(range(len(train_X)), batch_size)
            if i % 10 == 0:
                train_accuracy = sess.run(accuracy,feed_dict = {x:train_X[indeces_small], y_:train_Y[indeces_small], keep_prob:1})
                test_accuracy = sess.run(accuracy,feed_dict = {x:test_X, y_:test_Y, keep_prob:1})  
                print("experiment %d, step %d, train_accuracy %g, test_accuracy %g" %(file_index, i, train_accuracy, test_accuracy))               
                test_record.write(str(test_accuracy)+'\n')
            sess.run(train_step, feed_dict = {x:train_X[indeces_small], y_:train_Y[indeces_small], keep_prob:dropout_probability}) 
   
    test_record.close()


# In[14]:

## define hyper parameters
file_index = 1
random_seed = 30
training_size = 100
learning_rate = 0.00001
dropout_probability = 0.5
initialization = 'xavier'
max_interations = 1000
batch_size = 128

test_filename = './data/Benchmark_CNN/seed_'+str(random_seed)+                  '_training_size_'+str(training_size)+'_LR_'+str(learning_rate)+                  '_'+initialization+'_'+str(file_index)+'.txt'

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

generate_model(test_filename, train_data, train_label, mnist.test.images, mnist.test.labels)


# In[15]:

## define hyper parameters
file_index = 1
random_seed = 30
training_size = 100
learning_rate = 0.0001
dropout_probability = 0.5
initialization = 'xavier'
max_interations = 1000
batch_size = 128

test_filename = './data/Benchmark_CNN/seed_'+str(random_seed)+                  '_training_size_'+str(training_size)+'_LR_'+str(learning_rate)+                  '_'+initialization+'_'+str(file_index)+'.txt'

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

generate_model(test_filename, train_data, train_label, mnist.test.images, mnist.test.labels)


# In[16]:

## define hyper parameters
file_index = 1
random_seed = 30
training_size = 100
learning_rate = 0.001
dropout_probability = 0.5
initialization = 'xavier'
max_interations = 1000
batch_size = 128

test_filename = './data/Benchmark_CNN/seed_'+str(random_seed)+                  '_training_size_'+str(training_size)+'_LR_'+str(learning_rate)+                  '_'+initialization+'_'+str(file_index)+'.txt'

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

generate_model(test_filename, train_data, train_label, mnist.test.images, mnist.test.labels)


# In[17]:

## define hyper parameters
file_index = 1
random_seed = 30
training_size = 100
learning_rate = 0.01
dropout_probability = 0.5
initialization = 'xavier'
max_interations = 1000
batch_size = 128

test_filename = './data/Benchmark_CNN/seed_'+str(random_seed)+                  '_training_size_'+str(training_size)+'_LR_'+str(learning_rate)+                  '_'+initialization+'_'+str(file_index)+'.txt'

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

generate_model(test_filename, train_data, train_label, mnist.test.images, mnist.test.labels)


# In[18]:

## define hyper parameters
file_index = 1
random_seed = 30
training_size = 100
learning_rate = 0.1
dropout_probability = 0.5
initialization = 'xavier'
max_interations = 1000
batch_size = 128

test_filename = './data/Benchmark_CNN/seed_'+str(random_seed)+                  '_training_size_'+str(training_size)+'_LR_'+str(learning_rate)+                  '_'+initialization+'_'+str(file_index)+'.txt'

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

generate_model(test_filename, train_data, train_label, mnist.test.images, mnist.test.labels)


# In[19]:

## define hyper parameters
file_index = 1
random_seed = 30
training_size = 500
learning_rate = 0.1
dropout_probability = 0.5
initialization = 'xavier'
max_interations = 1000
batch_size = 128

test_filename = './data/Benchmark_CNN/seed_'+str(random_seed)+                  '_training_size_'+str(training_size)+'_LR_'+str(learning_rate)+                  '_'+initialization+'_'+str(file_index)+'.txt'

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

generate_model(test_filename, train_data, train_label, mnist.test.images, mnist.test.labels)


# In[20]:

## define hyper parameters
file_index = 1
random_seed = 30
training_size = 500
learning_rate = 0.01
dropout_probability = 0.5
initialization = 'xavier'
max_interations = 1000
batch_size = 128

test_filename = './data/Benchmark_CNN/seed_'+str(random_seed)+                  '_training_size_'+str(training_size)+'_LR_'+str(learning_rate)+                  '_'+initialization+'_'+str(file_index)+'.txt'

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

generate_model(test_filename, train_data, train_label, mnist.test.images, mnist.test.labels)


# In[21]:

## define hyper parameters
file_index = 1
random_seed = 30
training_size = 500
learning_rate = 0.001
dropout_probability = 0.5
initialization = 'xavier'
max_interations = 1000
batch_size = 128

test_filename = './data/Benchmark_CNN/seed_'+str(random_seed)+                  '_training_size_'+str(training_size)+'_LR_'+str(learning_rate)+                  '_'+initialization+'_'+str(file_index)+'.txt'

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

generate_model(test_filename, train_data, train_label, mnist.test.images, mnist.test.labels)


# In[12]:

## define hyper parameters
file_index = 1
random_seed = 30
training_size = 500
learning_rate = 0.0001
dropout_probability = 0.5
initialization = 'random'
max_interations = 1000
batch_size = 128

test_filename = './data/Benchmark_CNN/seed_'+str(random_seed)+                  '_training_size_'+str(training_size)+'_LR_'+str(learning_rate)+                  '_'+initialization+'_'+str(file_index)+'.txt'

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

generate_model(test_filename, train_data, train_label, mnist.test.images, mnist.test.labels)


# In[13]:

## define hyper parameters
file_index = 1
random_seed = 30
training_size = 500
learning_rate = 0.00001
dropout_probability = 0.5
initialization = 'random'
max_interations = 1000
batch_size = 128

test_filename = './data/Benchmark_CNN/seed_'+str(random_seed)+                  '_training_size_'+str(training_size)+'_LR_'+str(learning_rate)+                  '_'+initialization+'_'+str(file_index)+'.txt'

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

generate_model(test_filename, train_data, train_label, mnist.test.images, mnist.test.labels)


# In[22]:

## define hyper parameters
file_index = 1
random_seed = 30
training_size = 500
learning_rate = 0.0001
dropout_probability = 0.5
initialization = 'xavier'
max_interations = 1000
batch_size = 128

test_filename = './data/Benchmark_CNN/seed_'+str(random_seed)+                  '_training_size_'+str(training_size)+'_LR_'+str(learning_rate)+                  '_'+initialization+'_'+str(file_index)+'.txt'

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

generate_model(test_filename, train_data, train_label, mnist.test.images, mnist.test.labels)


# In[23]:

## define hyper parameters
file_index = 1
random_seed = 30
training_size = 500
learning_rate = 0.00001
dropout_probability = 0.5
initialization = 'xavier'
max_interations = 1000
batch_size = 128

test_filename = './data/Benchmark_CNN/seed_'+str(random_seed)+                  '_training_size_'+str(training_size)+'_LR_'+str(learning_rate)+                  '_'+initialization+'_'+str(file_index)+'.txt'

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

generate_model(test_filename, train_data, train_label, mnist.test.images, mnist.test.labels)


# In[ ]:



