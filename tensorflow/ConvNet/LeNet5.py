# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:06:11 2018

@author: admin
"""
import tensorflow as tf
import mnist_util
import os

BATCH_SIZE=128

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 6
CONV1_SIZE = 5
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 16
CONV2_SIZE = 5
# 全连接层的节点个数
FC1_SIZE= 120
FC2_SIZE= 84

NUM_LABELS = 10


REGULARAZTION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
BATCH_SIZE = 100
TRAINING_STEPS = 6000

TRAIN_IMAGES_PATH='E:/dataset/mnist/train-images-idx3-ubyte.gz'
TRAIN_LABELS_PATH='E:/dataset/mnist/train-labels-idx1-ubyte.gz'
TEST_IMAGES_PATH='E:/dataset/mnist/t10k-images-idx3-ubyte.gz'
TEST_LABELS_PATH='E:/dataset/mnist/t10k-labels-idx1-ubyte.gz'

class LeNet:
    
    def __init__(self,dataset):
        self.dataset=dataset
        self.train=False
        self.regularizer=None
        
    def _create_input(self):
        with tf.name_scope('input'):
            self.iterator=self.dataset.make_initializable_iterator()
            self.input,self.y=self.iterator.get_next()
            
    def _create_conv1(self):
        with tf.variable_scope('layer1-conv1'):
            self.conv1_weights=tf.get_variable("weight",
                                          [CONV1_SIZE,CONV1_SIZE,1,CONV1_DEEP],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.conv1_biases=tf.get_variable("bias",[CONV1_DEEP],
                                         initializer=tf.constant_initializer(0.0))
            # 使用边长为5，深度为32的过滤器，过滤器移动的步长为1，且使用全0填充
            self.conv1 = tf.nn.conv2d(self.input, self.conv1_weights, strides=[1, 1, 1, 1], padding='SAME',name='conv1')
            self.relu1 = tf.nn.relu(tf.nn.bias_add(self.conv1, self.conv1_biases),name='relu1')
            
    # 实现第二层池化层的前向传播过程。
    # 这里选用最大池化层，池化层过滤器的边长为2，SAME填充output=input/strides。
    # 这一层的输入是上一层的输出，也就是28*28*32的矩阵。输出为14*14*32的矩阵。
    def _create_pool1(self):
        with tf.name_scope('layer2-pool1'):
            self.pool1=tf.nn.max_pool(self.relu1,ksize=[1,2,2,1],strides=[1, 2, 2, 1], padding='SAME',name='pool1')
            
    # 声明第三层卷积层的变量并实现前向传播过程。
    # 这一层的输入为14*14*32的矩阵，输出为14*14*64的矩阵。        
    def _create_conv2(self):
        with tf.variable_scope('layer3-conv2'):
            self.conv2_weights=tf.get_variable('weight',
                                               [CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
            
            self.conv2_biases=tf.get_variable('bias',[CONV2_DEEP],
                                              initializer=tf.constant_initializer(0.0))
            self.conv2=tf.nn.conv2d(self.pool1,self.conv2_weights,strides=[1,1,1,1],padding='SAME',name='conv2')
            self.relu2=tf.nn.relu(tf.nn.bias_add(self.conv2,self.conv2_biases))
            
    # 实现第四层池化层的前向传播过程。
    # 这一层和第二层的结构是一样的。这一层的输入为14*14*64的矩阵，输出为7*7*64的矩阵。        
    def _create_pool2(self):
        with tf.name_scope('layer4-pool2'):
            self.pool2=tf.nn.max_pool(self.relu2,ksize=[1,2,2,1],strides=[1, 2, 2, 1], padding='SAME',name='pool2')
            
    def _reshape_pool2(self):
        # 将第四层池化层的输出转化为第五层全连接层的输入格式。
        # 第四层的输出为7*7*64的矩阵，然而第五层全连接层需要的输入格式为向量，所以在这里需要将这个7*7*64的矩阵拉直成一个向量。
        # pool2.get_shape函数可以得到第四层输出矩阵的维度而不需要手工计算。
        # 注意因为每一层神经网络的输入输出都为一个batch的矩阵，所以这里得到的维度也包含了一个batch中数据的个数。
        pool_shape = self.pool2.get_shape().as_list()
        # 计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长度及深度的乘积。
        # 注意这里pool_shape[0]为一个batch中样本的个数。
        self.nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        # 通过tf.reshape函数将第四层的输出变成一个batch的向量。
        self.reshaped = tf.reshape(self.pool2, [pool_shape[0], self.nodes])
        
    def _create_fc1(self):
        with tf.variable_scope('layer5-fc1'):
            self.fc1_weights=tf.get_variable('weight',[self.nodes,FC1_SIZE],
                                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        #if self.regularizer != None:
        #    tf.add_to_collection('losses', self.regularizer(self.fc1_weights))
        fc1_biases = tf.get_variable('bias', [FC1_SIZE], initializer = tf.constant_initializer(0.1))
        self.fc1 = tf.nn.relu(tf.matmul(self.reshaped, self.fc1_weights)+fc1_biases)
        #if self.train:
        #    self.fc1 = tf.nn.dropout(self.fc1, 0.5)

    def _create_fc2(self):
         with tf.variable_scope('layer6-fc2'):
            self.fc2_weights = tf.get_variable("weight", [FC1_SIZE, FC2_SIZE],
                                          initializer = tf.truncated_normal_initializer(stddev = 0.1))
            #if self.regularizer != None:
            #    tf.add_to_collection('losses', self.regularizer(self.fc2_weights))
            self.fc2_biases = tf.get_variable('bias', [FC2_SIZE], initializer = tf.constant_initializer(0.1))
            self.fc2 = tf.matmul(self.fc1, self.fc2_weights) + self.fc2_biases
            
    def _create_fc3(self):
        with tf.variable_scope('layer7-fc3'):
            self.fc3_weights = tf.get_variable("weight", [FC2_SIZE, NUM_LABELS],
                                          initializer = tf.truncated_normal_initializer(stddev = 0.1))
            if self.regularizer != None:
                tf.add_to_collection('losses', self.regularizer(self.fc3_weights))
            self.fc3_biases = tf.get_variable('bias', [NUM_LABELS], initializer = tf.constant_initializer(0.1))
            self.fc3 = tf.matmul(self.fc2, self.fc3_weights) + self.fc3_biases
        return self.fc3
    
    def create_network(self):
        self._create_input()
        self._create_conv1()
        self._create_pool1()
        self._create_conv2()
        self._create_pool2()
        self._reshape_pool2()
        self._create_fc1()
        self._create_fc2()
        output=self._create_fc3()
        return output
    
    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()
    
    def trainMnist(self):
        self.train=True
        self.regularizer=tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
        #神经网络的计算输出
        output=self.create_network()
        self.global_step = tf.Variable(0, trainable=False)
        #定义损失函数、学习率、滑动平均操作以及训练过程
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, self.global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=tf.argmax(self.y, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        self.loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        self.learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, self.global_step,BATCH_SIZE, LEARNING_RATE_DECAY)
        train_step = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            self.train_op = tf.no_op(name='train')
        # 初始化TensorFlow持久化类。
        try:
            os.mkdir('checkpoints')
        except FileExistsError:
            pass
        saver = tf.train.Saver()
        self._create_summaries()
        with tf.Session() as sess:
            sess.run(self.iterator.initializer)
            sess.run(tf.global_variables_initializer())            
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
            writer = tf.summary.FileWriter('graphs', sess.graph)
            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                
            for index in range(TRAINING_STEPS):
                try:
                    _,loss_value, step,summary= sess.run([self.train_op, self.loss, self.global_step,self.summary_op])
                    writer.add_summary(summary, global_step=index)
                    if index % 100 == 0:
                        print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                        saver.save(sess, 'checkpoints/skip-gram', index)
                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)
            writer.close()
    
    def test_accuracy(self):
        output=self.create_network()
        predict=tf.nn.softmax(output)
        correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        mean_accuracy=0.0
        step=1
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.iterator.initializer)
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            while True:
                try:
                    step_accuracy,=sess.run([accuracy])
                    mean_accuracy=(step-1)/step*mean_accuracy+1/step*step_accuracy
                    print("current accuracy:",step_accuracy)
                    print("total accuracy:",mean_accuracy)
                    step+=1
                except tf.errors.OutOfRangeError:
                    break
                

def train_data_batch_gen():
    yield from mnist_util.batch_gen(BATCH_SIZE,TRAIN_IMAGES_PATH,TRAIN_LABELS_PATH)

def test_data_batch_gen():
    yield from mnist_util.batch_gen(BATCH_SIZE,TEST_IMAGES_PATH,TEST_LABELS_PATH)
    
def train_mnist():
    tf.reset_default_graph()
    dataset=tf.data.Dataset.from_generator(train_data_batch_gen,
                                (tf.float32, tf.float32), 
                                (tf.TensorShape([BATCH_SIZE,mnist_util.IMAGE_SIZE,mnist_util.IMAGE_SIZE,1]),
                                 tf.TensorShape([BATCH_SIZE,10])))
    leNet=LeNet(dataset)
    leNet.trainMnist()
    
def test_mnist():
    tf.reset_default_graph()
    dataset=tf.data.Dataset.from_generator(test_data_batch_gen,
                                (tf.float32, tf.float32), 
                                (tf.TensorShape([BATCH_SIZE,mnist_util.IMAGE_SIZE,mnist_util.IMAGE_SIZE,1]),
                                 tf.TensorShape([BATCH_SIZE,10])))
    leNet=LeNet(dataset)
    leNet.test_accuracy()
    
def main():
    train_mnist()
    test_mnist()
    
if __name__=="__main__":
    main()