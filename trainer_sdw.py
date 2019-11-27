from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import time
import math
import os


def train():
    ##### load img #####
    path = 'img/_pre/'
    filenames = os.listdir(path)
    label = []
    data = []
    size = 80

    for filename in filenames:
        label.append(filename[0])
        img = cv2.imread(path + filename)
        img = cv2.resize(img, (size, size))
        img = img / 255
        data.append(img)
    ### load img end ###

    ##### init var #####
    x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=0.6, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=42)

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    enc = OneHotEncoder(categories='auto')

    y_train = y_train.reshape([-1, 1])
    y_train = enc.fit_transform(y_train).toarray()

    y_val = y_val.reshape([-1, 1])
    y_val = enc.fit_transform(y_val).toarray()

    y_test = y_test.reshape([-1, 1])
    y_test = enc.fit_transform(y_test).toarray()

    tf.reset_default_graph() # 변수들의 컨텍스트가 유지되는 주피터 노트북에서 필요
    tf.set_random_seed(777)
    ### init var end ###

    ##### CNN #####
    stddev = 0.01
    #80*80*3
    X = tf.placeholder(tf.float32, shape=[None, size, size, 3], name='x')
    Y = tf.placeholder(tf.float32, [None, 10], name='y')

    #40*40*32
    W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=stddev))
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    #20*20*64
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=stddev))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    #10*10*128
    W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=stddev))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    #5*5*256
    W4 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=stddev))
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
    ### CNN end ###

    ##### init #####
    learning_rate = 0.001
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accr')

    keep_prob = tf.placeholder(tf.float32)
    saver = tf.train.Saver()

    total_size = len(x_train)
    batch_size = 2048

    loss_train_list = []
    loss_val_list = []
    accr_val_list = []
    ### init end ###

    ##### Early stop #####
    epoch = 0      # Epoch count
    patient = 6    # Early stopping condition
    stop_count = 0 # Early stop count
    min_loss_val = None
    max_accr_val = None
    ### Early stop end ###

    ##### train #####
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        start = time.time()

        while True:
            epoch += 1

            try:
                for i in range(math.ceil(total_size / batch_size)):
                    batch_x_data = x_train[batch_size * i:batch_size * (i + 1)]
                    batch_y_data = y_train[batch_size * i:batch_size * (i + 1)]
                    loss_train, _ = sess.run([cost, optimizer], feed_dict={X:batch_x_data, Y:batch_y_data, keep_prob:0.5})
            except:
                if batch_size != 0:
                    print("batch_size from {}".format(batch_size), end=" ")
                    batch_size //= 2
                    print("to {}".format(batch_size))
                    continue
                else:
                    raise

            accr_val, loss_val = sess.run([accuracy, cost], feed_dict={X:x_val, Y:y_val})

            if min_loss_val is None:
                min_loss_val = loss_val
            else:
                if loss_val >= min_loss_val:
                    stop_count += 1
                else:
                    min_loss_val = loss_val
                    stop_count = 0

            if max_accr_val is None:
                max_accr_val = accr_val
            else:
                if accr_val > max_accr_val:
                    max_accr_val = accr_val
                    if accr_val >= 0.9:
                        save_file = './model/{:.2f}%_{:.4f}.ckpt'.format(accr_val * 100, loss_val)
                        saver.save(sess, save_file)

            if stop_count <= patient:
                print('epoch = {} | loss = {:.4f} | accr = {:.2f}% | count = {}'.format(epoch, loss_val, accr_val * 100, stop_count))
            else:
                print('===== Learning Finished =====')
                print('time = {}s'.format(time.time() - start))
                break

            loss_train_list.append(loss_train)
            loss_val_list.append(loss_val)
            accr_val_list.append(accr_val)

        acc_v = sess.run(accuracy, feed_dict={X:x_test, Y:y_test})
        # print(tf.argmax(a))
        print("Accr :{:2.2f}%".format(acc_v * 100))
    ### train end ###

    ##### Plot #####
    lenGraph = len(loss_train_list)
    plt.plot(range(lenGraph), loss_train_list)
    plt.plot(range(lenGraph), loss_val_list, c ='r')
    plt.plot(range(lenGraph), accr_val_list, c ='g')
    plt.ylim(0, 1)
    plt.xlim(0, lenGraph)
    plt.tight_layout()
    plt.show()
    ### Plot end ###

def load():
    ##### load img #####
    path = 'img/_test/'
    filenames = os.listdir(path)
    label = []
    data = []
    size = 80

    for filename in filenames:
        label.append(filename[0])
        img = cv2.imread(path + filename)
        img = cv2.resize(img, (size, size))
        img = img / 255
        data.append(img)
    ### load img end ###


    enc = OneHotEncoder(categories='auto')
    label = np.array(label)
    label = label.reshape([-1, 1])
    label = enc.fit_transform(label).toarray()

    ##### Load model #####
    with tf.Session() as sess:
        start = time.time()
        saver = tf.train.import_meta_graph('model/92.99%_0.3473.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('model/'))
        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        feed_dict = {x:data, y:label}

        loss = graph.get_tensor_by_name("loss:0")
        accr = graph.get_tensor_by_name("accr:0")

        loss, accr = sess.run([loss, accr], feed_dict)
        print('loss = {:.4f} | accr = {:.2f}% | time = {}s'.format(loss, accr * 100, time.time() - start))
    ### Load model end ###
