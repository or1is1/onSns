import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# def train():
path = 'img/_train/'
filenames = os.listdir(path)

columns = ['filename', 'width', 'height']
df = pd.DataFrame(columns=columns)
label = []
data = []
size = 80

for filename in filenames:
    label.append(filename[0])
    img = cv2.imread(path + filename) # 4 color
#     img = cv2.imread(path + filename, cv2.IMREAD_GRAYSCALE) # 4 Gray
    img = cv2.resize(img, (size, size))
#     img = img.reshape(size, size, 1) # 4 Gray
    img = img / 255
    data.append(img)
    
start = time.time()

X_train, X_test, y_train, y_test = train_test_split(data, label)

y_train = np.array(y_train)
y_test = np.array(y_test)

y_train=y_train.reshape([-1,1])
enc=OneHotEncoder()
enc.fit(y_train)
y_train=enc.transform(y_train).toarray()

y_test=y_test.reshape([-1,1])
enc=OneHotEncoder()
enc.fit(y_test)
y_test=enc.transform(y_test).toarray()

tf.reset_default_graph()
tf.set_random_seed(777)  # reproducibility

learning_rate = 0.01
training_epochs = 100
batch_size = 100
keep_prob = tf.placeholder(tf.float32)

#80*80*3
# X = tf.placeholder(tf.float32, shape=[None, size, size, 1]) # 4 Gray
X = tf.placeholder(tf.float32, shape=[None, size, size, 3]) # 4 color
Y = tf.placeholder(tf.float32, [None, 10])

#40*40*32
# W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.5)) # 4 Gray
W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.5)) # 4 color
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

#20*20*64
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.5))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

#10*10*128
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.5))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

#5*5*256
W4 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.5))
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

L1_flat = tf.reshape(L4, [-1, 5 * 5 * 256])
W1_flat = tf.Variable(tf.random_normal([5 * 5 * 256, 5 * 5 * 256]))
B1_flat = tf.Variable(tf.random_normal([5 * 5 * 256]))
L1_flat = tf.matmul(L1_flat, W1_flat) + B1_flat

W2_flat = tf.Variable(tf.random_normal([5 * 5 * 256, 10]))
B2_flat = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L1_flat, W2_flat) + B2_flat

########################################################################
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# dataset = dataset.batch(50) #배치사이즈
# iterator = dataset.make_initializable_iterator() #함수 호출, 
# x_epoch, y_epoch = iterator.get_next() #next batch

sess = tf.Session()
sess.run(tf.global_variables_initializer())

tmplst2 = []

step = 0
count = 0
min_loss_test = None

# for epoch in range(training_epochs):
while True:
#     if step == 10:
#         break
    step += 1
    loss_train, _ = sess.run([cost, optimizer], feed_dict = {X:X_train, Y:y_train, keep_prob: 1.0})
    loss_test = sess.run(cost, feed_dict = {X: X_test, Y: y_test, keep_prob: 1.0})
    
    if min_loss_test is None:
        min_loss_test = loss_test
    elif loss_test >= min_loss_test:
        count += 1
    else:
        min_loss_test = loss_test
        count = 0

    tmplst2.append(loss_test)
    print('===== step : {} ====='.format(step))
    print('train cost = {}'.format(loss_train))
    print('test cost = {}'.format(loss_test))
    print('count = {}'.format(count))
    
    if count == 1:
        pass # TODO saveModel()
    elif count >= 4:
        print('time = {}s'.format(time.time() - start))
        print('===== Learning Finished =====')
        break

acc_v = sess.run(accuracy, feed_dict={X:X_test, Y:y_test, keep_prob: 1.0})
# print(tf.argmax(a))
print("Accr :{:2.2f}%".format(acc_v * 100))

if len(tmplst2) > 10:
    tmplst2 = tmplst2[-10:]

plt.plot(range(len(tmplst2)), tmplst2, c ='r')
plt.show()

# train()
