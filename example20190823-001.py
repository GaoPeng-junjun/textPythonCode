# -*- coding: utf-8 -*-


import tensorflow as tf
# 导入input_data用于自动下载和安装MNIST数据集
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import matplotlib.pyplot as plt
import cv2 as cv


mnist = read_data_sets("MNIST_data/", one_hot=True)

# 创建两个占位符，x为输入网络的图像，y_为输入网络的图像类别
x = tf.compat.v1.placeholder("float", shape=[None, 784])
y_ = tf.compat.v1.placeholder("float", shape=[None, 10])


# 权重初始化函数
def weight_variable(shape):
    # 输出服从截尾正态分布的随机值
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置初始化函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 创建卷积op
# x 是一个4维张量，shape为[batch,height,width,channels]
# 卷积核移动步长为1。填充类型为SAME,可以不丢弃任何像素点, VALID丢弃边缘像素点
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# 创建池化op
# 采用最大池化，也就是取窗口中的最大值作为结果
# x 是一个4维张量，shape为[batch,height,width,channels]
# ksize表示pool窗口大小为2x2,也就是高2，宽2
# strides，表示在height和width维度上的步长都为2
def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 第1层，卷积层
# 初始化W为[5,5,1,32]的张量，表示卷积核大小为5*5，1表示图像通道数，6表示卷积核个数即输出6个特征图
W_conv1 = weight_variable([5, 5, 1, 6])
# 初始化b为[6],即输出大小
b_conv1 = bias_variable([6])

# 把输入x(二维张量,shape为[batch, 784])变成4d的x_image，x_image的shape应该是[batch,28,28,1]
# -1表示自动推测这个维度的size
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 把x_image和权重进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max_pooling
# h_pool1的输出即为第一层网络输出，shape为[batch,14,14,6]
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第2层，卷积层
# 卷积核大小依然是5*5，通道数为6，卷积核个数为16
W_conv2 = weight_variable([5, 5, 6, 16])
b_conv2 = weight_variable([16])

# h_pool2即为第二层网络输出，shape为[batch,7,7,16]
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第3层, 全连接层
# 这层是拥有1024个神经元的全连接层
# W的第1维size为7*7*16，7*7是h_pool2输出的size，16是第2层输出神经元个数
W_fc1 = weight_variable([7*7*16, 120])
b_fc1 = bias_variable([120])

# 计算前需要把第2层的输出reshape成[batch, 7*7*16]的张量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#  Dropout层
#  为了减少过拟合，在输出层前加入dropout
keep_prob = tf.compat.v1.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#  输出层
#  最后，添加一个softmax层
#  可以理解为另一个全连接层，只不过输出时使用softmax将网络输出值转换成了概率
W_fc2 = weight_variable([120, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#  预测值和真实值之间的交叉墒
cross_entropy = -tf.reduce_sum(y_ * tf.math.log(y_conv))

#  train op, 使用ADAM优化器来做梯度下降。学习率为0.0001
train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#  评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。
#  因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置
correct_predict = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
predict_lables = tf.argmax(y_conv, 1)
#  计算正确预测项的比例，因为tf.equal返回的是布尔值，
#  使用tf.cast把布尔值转换成浮点数，然后用tf.reduce_mean求平均值
accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))

saver = tf.compat.v1.train.Saver()


#  开始训练模型，循环20000次，每次随机从训练集中抓取50幅图像
def cnn_train():
    # 创建一个交互式Session
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            # 每100次输出一次日志
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            saver.save(sess, 'model')
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


#预测
def predict():
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver(tf.global_variables())
    saver.restore(sess, 'model')
    print("test accuracy %g" % accuracy.eval(
        feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# cnn_train()

sess = tf.compat.v1.InteractiveSession()
# sess.run(tf.compat.v1.global_variables_initializer())
# # 读入二进制文件
# image_raw = tf.gfile.FastGFile('test2.png', 'rb').read()
# # 解码为tf中的图像格式
# img = tf.image.decode_jpeg(image_raw)  # Tensor
# img_gray = sess.run(tf.image.rgb_to_grayscale(img))
# print(img.shape)
# img_ = img_gray.eval()
image = cv.imread("test2.png")
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
gray = gray[34:62, 48:76]
# print(gray.shape)
img_ = sess.run(tf.reshape(gray, [28*28, ]))
# print(img_)
for i in range(len(img_)):
    if img_[i] > 175:
        img_[i] = 255
img_normal = [(255 - x) / 255 for x in img_]
# minst_ = sess.run(tf.reshape(img_, [28, 28]))
# print(len(img_normal))

# minst_ = [int(255 - 255*x) for x in mnist.test.images[0, :]]
minst_ = sess.run(tf.reshape(img_, [28, 28]))

# print(mnist.test.images.shape)
img_normal = sess.run(tf.reshape(img_normal, [1, 28*28]))

# img_normal = sess.run(tf.reshape(mnist.test.images[0, :], [1, 28*28]))
label = sess.run(tf.reshape([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], [1, 10]))
# # print(minst_img)

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
# img_ = tf.reshape(gray, [28*28, 1])
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
saver.restore(sess, 'model')
print("test lable %g" % predict_lables.eval(
        feed_dict={x: img_normal, y_: label, keep_prob: 1.0}))
# print("test accuracy %g" % accuracy.eval(
#         feed_dict={x: img_normal, y_: label, keep_prob: 1.0}))


plt.figure(1)
plt.imshow(minst_)
plt.show()

# predict()
