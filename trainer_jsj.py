#!/usr/bin/env python
# coding: utf-8

# ## CNN

# ### 데이터 : 화장품 이미지 + 모델 저장

# In[ ]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


get_ipython().run_line_magic('matplotlib', 'inline')


def model(img, patience):

    img = os.listdir(img)
    
    c=[]
    for i in range(len(img)):
        b=cv2.imread('img/_pre/{}'.format(img[i]))
        b=cv2.resize(b, dsize=(50, 50))
        b = b / 255
        c.append(b)
    
    label=[]
    for i in range(len(img)):
        label.append(img[i][0])
    
    
    X_train, X_val, y_train, y_val = train_test_split(c, label, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    from sklearn.preprocessing import OneHotEncoder

    y_train=y_train.reshape([-1,1])
    enc=OneHotEncoder()
    enc.fit(y_train)
    y_train=enc.transform(y_train).toarray()

    y_val=y_val.reshape([-1,1])
    enc=OneHotEncoder()
    enc.fit(y_val)
    y_val=enc.transform(y_val).toarray()

    y_test=y_test.reshape([-1,1])
    enc=OneHotEncoder()
    enc.fit(y_test)
    y_test=enc.transform(y_test).toarray()

    import tensorflow as tf
    import random
    import matplotlib.pyplot as plt

    tf.reset_default_graph()
    tf.set_random_seed(777)  # reproducibility

    learning_rate = 0.0001
    training_epochs = 30
    batch_size = 100
    keep_prob = tf.placeholder(tf.float32)

    X = tf.placeholder( tf.float32, shape=[None, 50,50,3], name='X')  
    Y = tf.placeholder(tf.float32, [None, 10], name='Y')

    #50*50*3
    W1 = tf.Variable(tf.random_normal([5, 5, 3, 16]), name='W1')
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    #25*25*16
    W2 = tf.Variable(tf.random_normal([5, 5, 16, 32]), name='W2')
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    #13*13*32

    W3 = tf.Variable(tf.random_normal([5, 5, 32, 64]), name='W3')
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    W4 = tf.Variable(tf.random_normal([5, 5, 64, 128]), name='W4')
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    L3_flat = tf.reshape(L4, [-1, 4 * 4 * 128])
    #7*7*64

    W5 = tf.Variable(tf.random_normal([4 * 4 * 128, 10], stddev=0.01), name='W5')
    b = tf.Variable(tf.random_normal([10]))
    logits = tf.matmul(L3_flat, W5) + b

    ########################################################################
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.batch(batch_size) #배치사이즈
    iterator = dataset.make_initializable_iterator() #함수 호출, 
    x_epoch, y_epoch = iterator.get_next() #next batch
    step = 0

    # tf.train.Saver를 이용해서 모델과 파라미터를 저장합니다.
    save_file = './model/model.ckpt'
    saver = tf.train.Saver()

    with tf.Session() as sess:
        #haha = tf.summary.FileWriter('/tmp/1', sess.graph)
        sess.run(tf.global_variables_initializer())
        tmplst1 = []
        tmplst2 = []
        a = []    
        prv_absSub = 0
        lossMin = 0
        epoch = 0
        count = 0
        prv = None

        while True:
            sess.run(iterator.initializer) #iterator 상태를 처음 초기화하
            epoch += 1
            avg_loss = 0

            while True:
                try:
                    batch_x_data, batch_y_data = sess.run([x_epoch, y_epoch])
                    loss_v, _ = sess.run([cost,optimizer], feed_dict={X:batch_x_data,Y:batch_y_data, keep_prob: 1.0}) # 드롭아웃 일단 뺌
                    #loss_x = sess.run(cost, feed_dict={X:X_test,Y:y_test, keep_prob: 1.0})

                    step = step +100

                except tf.errors.OutOfRangeError:
                    step = 0
                    break

            loss_x = sess.run(cost, feed_dict = {X:X_val, Y:y_val, keep_prob: 1.0})
            acc_v = sess.run(accuracy, feed_dict={X:X_val,Y:y_val, keep_prob: 1.0})
            
            
            absSub = abs(loss_v - loss_x)
            
            if epoch == 1:
                lossMin = loss_x
                prv_absSub = abs(loss_v - loss_x)
            else:
                if loss_x > lossMin:
                    count += 4
                else:
                    lossMin = loss_x
                    count = 0
                
                if absSub > prv_absSub:
                    count += 1
                else:
                    prv_absSub = absSub
                    
                    if count > 0:
                        count -= 2

#             print('epoch :', epoch, '|', 'loss_x :', loss_x, '|', 'abs(avg_loss - loss_x) :', abs(loss_v - loss_x), '|', 'count :', count)
            print('epoch :', epoch, '|', 'loss_x :', loss_x, '|', 'count :', count)
            a.append(abs(loss_v - loss_x))

            tmplst1.append(loss_v)
            tmplst2.append(loss_x)

            if count >= patience * 3:
                break
            
        saver.save(sess, save_file)
        acc_v = sess.run(accuracy, feed_dict={X:X_test,Y:y_test, keep_prob: 1.0})
        print(tf.argmax(a))
        print('acc_v :', acc_v)

        plt.plot(range(len(tmplst1)), tmplst1)
        plt.plot(range(len(tmplst2)), tmplst2, c ='r')
        plt.show()
        
add1 = "img/_pre/"
patience = 8

model(add1, patience)


# ### 위의 모델 불러오기

# In[ ]:


def loadModel(img, patience):

    img = os.listdir(img)
    
    c=[]
    for i in range(len(img)):
        b=cv2.imread('img/_pre/{}'.format(img[i]))
        b=cv2.resize(b, dsize=(50, 50))
        c.append(b)
    
    label=[]
    for i in range(len(img)):
        label.append(img[i][0])
    
    
    X_train, X_val, y_train, y_val = train_test_split(c, label, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(c, label, test_size=0.5, random_state=42)

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    from sklearn.preprocessing import OneHotEncoder

    y_train=y_train.reshape([-1,1])
    enc=OneHotEncoder()
    enc.fit(y_train)
    y_train=enc.transform(y_train).toarray()

    y_val=y_val.reshape([-1,1])
    enc=OneHotEncoder()
    enc.fit(y_val)
    y_val=enc.transform(y_val).toarray()

    y_test=y_test.reshape([-1,1])
    enc=OneHotEncoder()
    enc.fit(y_test)
    y_test=enc.transform(y_test).toarray()

    import tensorflow as tf
    import random
    import matplotlib.pyplot as plt

    tf.reset_default_graph()
    tf.set_random_seed(777)  # reproducibility

    learning_rate = 0.0001
    training_epochs = 30
    batch_size = 100
    keep_prob = tf.placeholder(tf.float32)

    X = tf.placeholder(tf.float32, shape=[None, 50,50,3], name='X')  
    Y = tf.placeholder(tf.float32, [None, 10], name='Y')

    #50*50*3
    W1 = tf.Variable(tf.random_normal([5, 5, 3, 16], stddev=0.01), name='W1')
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    #25*25*16
    W2 = tf.Variable(tf.random_normal([5, 5, 16, 32], stddev=0.01), name='W2')
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    #13*13*32

    W3 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01), name='W3')
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    W4 = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev=0.01), name='W4')
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    L3_flat = tf.reshape(L4, [-1, 4 * 4 * 128])
    #7*7*64

    W5 = tf.Variable(tf.random_normal([4 * 4 * 128, 10], stddev=0.01), name='W5')
    b = tf.Variable(tf.random_normal([10]))
    logits = tf.matmul(L3_flat, W5) + b

    ########################################################################
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.batch(batch_size) #배치사이즈
    iterator = dataset.make_initializable_iterator() #함수 호출, 
    x_epoch, y_epoch = iterator.get_next() #next batch
    step = 0

    # tf.train.Saver를 이용해서 모델과 파라미터를 저장합니다.
    save_file = './model/model.ckpt'
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, save_file)

        acc = sess.run(accuracy, feed_dict = {X:X_test, Y:y_test, keep_prob: 1.0})

        print(acc)
        
add1 = "img/_pre/"
patience = 5

loadModel(add1, patience)

# ### 데이터 전처리 + Early stopping
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


%matplotlib inline


def model(img, patience, imgCount):     # 경로, patience, 클래스당 이미지 갯수

    c, label = define(img, imgCount)
        
    X_train, X_val, y_train, y_val = train_test_split(c, label, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    from sklearn.preprocessing import OneHotEncoder

    y_train=y_train.reshape([-1,1])
    enc=OneHotEncoder()
    enc.fit(y_train)
    y_train=enc.transform(y_train).toarray()

    y_val=y_val.reshape([-1,1])
    enc=OneHotEncoder()
    enc.fit(y_val)
    y_val=enc.transform(y_val).toarray()

    y_test=y_test.reshape([-1,1])
    enc=OneHotEncoder()
    enc.fit(y_test)
    y_test=enc.transform(y_test).toarray()

    import tensorflow as tf
    import random
    import matplotlib.pyplot as plt

    tf.reset_default_graph()
    tf.set_random_seed(777)  # reproducibility

    learning_rate = 0.001
    training_epochs = 30
    batch_size = 256
    keep_prob = tf.placeholder(tf.float32)

    X = tf.placeholder( tf.float32, shape=[None, 50,50,3], name='X')  
    Y = tf.placeholder(tf.float32, [None, 10], name='Y')

    #50*50*3
    W1 = tf.Variable(tf.random_normal([5, 5, 3, 16]), name='W1')
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    #25*25*16
    W2 = tf.Variable(tf.random_normal([5, 5, 16, 32]), name='W2')
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    #13*13*32

    W3 = tf.Variable(tf.random_normal([5, 5, 32, 64]), name='W3')
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    W4 = tf.Variable(tf.random_normal([5, 5, 64, 128]), name='W4')
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    L3_flat = tf.reshape(L4, [-1, 4 * 4 * 128])
    #7*7*64

    W5 = tf.Variable(tf.random_normal([4 * 4 * 128, 10], stddev=0.01), name='W5')
    b = tf.Variable(tf.random_normal([10]))
    logits = tf.matmul(L3_flat, W5) + b

    ########################################################################
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.batch(batch_size) #배치사이즈
    iterator = dataset.make_initializable_iterator() #함수 호출, 
    x_epoch, y_epoch = iterator.get_next() #next batch
    
    step = 0

    # tf.train.Saver를 이용해서 모델과 파라미터를 저장합니다.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        #haha = tf.summary.FileWriter('/tmp/1', sess.graph)
        sess.run(tf.global_variables_initializer())
        tmplst1 = []
        tmplst2 = []
        a = []    
        prv_absSub = 0
        lossMin = 0
        epoch = 0
        count = 0
        lr_count = 0
        prv = None

        while True:
            sess.run(iterator.initializer) #iterator 상태를 처음 초기화하
            epoch += 1
            avg_loss = 0

            while True:
                try:
                    batch_x_data, batch_y_data = sess.run([x_epoch, y_epoch])
                    loss_v, _ = sess.run([cost,optimizer], feed_dict={X:batch_x_data,Y:batch_y_data, keep_prob: 1.0}) # 드롭아웃 일단 뺌
                    
                    step += batch_size

                except tf.errors.OutOfRangeError:
                    step = 0
                    break
            
            loss_x = sess.run(cost, feed_dict = {X:X_val, Y:y_val, keep_prob: 1.0})
            acc_v = sess.run(accuracy, feed_dict={X:X_val,Y:y_val, keep_prob: 1.0})
                    
            absSub = abs(loss_v - loss_x)
            
            if epoch == 1:
                lossMin = loss_x
                prv_absSub = abs(loss_v - loss_x)
            else:
                if loss_x > lossMin:
                    count += 1
                else:
                    lossMin = loss_x
                    count = 0
                    save_file = './model/{}_{}_{}.ckpt'.format(epoch, lossMin, acc_v)
                    saver.save(sess, save_file)
                
#                 if absSub > prv_absSub:
#                     count += 1

#                 else:
#                     prv_absSub = absSub
                    
#                     if count > 0:
#                         count -= 2

            print('epoch :', epoch, '| loss_x :', loss_x, '| count :', count, '| accr :', acc_v, '| lr :', learning_rate)
            a.append(abs(loss_v - loss_x))

            tmplst1.append(loss_v)
            tmplst2.append(loss_x)

            if count >= patience:
                break
                           
            if count == patience // 2 and lr_count < 3:
                learning_rate /= 10
                count = 0
                lr_count += 1
            
        acc_v = sess.run(accuracy, feed_dict={X:X_test,Y:y_test, keep_prob: 1.0})
        print(tf.argmax(a))
        print('acc_v :', acc_v)

        plt.plot(range(len(tmplst1)), tmplst1)
        plt.plot(range(len(tmplst2)), tmplst2, c ='r')
        plt.show()
        
        
add1 = "img/_pre/"
patience = 8

model(add1, patience, 300)
