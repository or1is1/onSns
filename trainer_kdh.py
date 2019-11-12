import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def train():
    start = time.time()
    print("train start!")
    path = 'img/_pre/'
    filenames = os.listdir(path)
    
    label = []
    c = []
    
    for file in filenames:
        label.append(file[0])
        a = cv2.imread(path + file)
        a = cv2.resize(a, dsize=(50, 50))
        c.append(a)
    
    X_train, X_test, y_train, y_test = train_test_split(c, label)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
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

    learning_rate = 0.001
    training_epochs = 10
    batch_size = 100
    keep_prob = tf.placeholder(tf.float32)

    X = tf.placeholder(tf.float32, shape=[None, 50,50,3])  
    Y = tf.placeholder(tf.float32, [None, 10])
        
    #50*50*3
    W1 = tf.Variable(tf.random_normal([7, 7, 3, 16], stddev=0.01))
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
    

    #25*25*16
    W2 = tf.Variable(tf.random_normal([7, 7, 16, 32], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    
    #13*13*32
    W3 = tf.Variable(tf.random_normal([7, 7, 32, 64], stddev=0.01))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    
    #7*7*64
    W4 = tf.Variable(tf.random_normal([7, 7, 64, 128], stddev=0.01))
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    L3_flat = tf.reshape(L4, [-1, 4 * 4 * 128])
    
    W5 = tf.Variable(tf.random_normal([4 * 4 * 128, 10],stddev=0.01))
    
    b = tf.Variable(tf.random_normal([10]))
    logits = tf.matmul(L3_flat, W5) + b

    ########################################################################
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # dataset = dataset.batch(50) #배치사이즈
    # iterator = dataset.make_initializable_iterator() #함수 호출, 
    # x_epoch, y_epoch = iterator.get_next() #next batch
    step = 0

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    tmplst1 = []
    tmplst2 = []
        
    for epoch in range(training_epochs):
        c, _ = sess.run([cost, optimizer], feed_dict = {X:X_train, Y:y_train, keep_prob: 1.0})
        loss_x = sess.run(cost, feed_dict = {X: X_test, Y: y_test, keep_prob: 1.0})
       
        tmplst1.append(c)
        tmplst2.append(loss_x)
        print('===== Epoch :', epoch + 1, ' =====')
        print('train cost =', '{:.9f}'.format(c))
        print('test cost =', '{:.9f}'.format(loss_x))
    print('===== Learning Finished =====')
        
    
    acc_v = sess.run(accuracy, feed_dict={X:X_test, Y:y_test, keep_prob: 1.0})
    # print(tf.argmax(a))
    print("Accr :", acc_v)
    print("time :", time.time() - start)

    plt.plot(range(len(tmplst1)), tmplst1)
    plt.plot(range(len(tmplst2)), tmplst2, c ='r')
    plt.ylim(0, 3)
    plt.show()

train()