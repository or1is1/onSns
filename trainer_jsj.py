#!/usr/bin/env python
# coding: utf-8

# ## CNN

# ### 전체데이터

# In[ ]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping


def train():
    start = time.time()
    print("train start!")
    path = 'C:\Users\edu\img\_pre'
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

        if  EarlyStopping(patience=10, verbose=1).validate(c):
            break
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


# ### Early stopping(기본)

# In[ ]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


get_ipython().run_line_magic('matplotlib', 'inline')


def model(NARS, YSL):
    NARS_data = os.listdir(NARS)
    print(len(NARS_data))
    YSL_data = os.listdir(YSL)
    print(len(YSL_data))

    c=[]
    for i in range(len(NARS_data)):
        a=cv2.imread('./cosmetics/NARS/{}'.format(NARS_data[i]))
        a=cv2.resize(a, dsize=(50, 50))
        c.append(a)

    for i in range(len(YSL_data)):
        b=cv2.imread('./cosmetics/YSL/{}'.format(YSL_data[i]))
        b=cv2.resize(b, dsize=(50, 50))
        c.append(b)

    label=[]
    for i in range(len(NARS_data)):
        label.append(1)


    for i in range(len(YSL_data)):
        label.append(0)

    X_train, X_test, y_train, y_test = train_test_split(c, label, test_size=0.33, random_state=42)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    from sklearn.preprocessing import OneHotEncoder

    y_train=y_train.reshape([-1,1])
    enc=OneHotEncoder()
    enc.fit(y_train)
    y_train=enc.transform(y_train).toarray()
    
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

    X = tf.placeholder(tf.float32, shape=[None, 50,50,3])  
    Y = tf.placeholder(tf.float32, [None, 2])

    #50*50*3
    W1 = tf.Variable(tf.random_normal([5, 5, 3, 16], stddev=0.01))
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    #25*25*16
    W2 = tf.Variable(tf.random_normal([5, 5, 16, 32], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    #13*13*32

    W3 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    W4 = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev=0.01))
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    L3_flat = tf.reshape(L4, [-1, 4 * 4 * 128])
    #7*7*64

    W5 = tf.Variable(tf.random_normal([4 * 4 * 128, 2],stddev=0.01))
    b = tf.Variable(tf.random_normal([2]))
    logits = tf.matmul(L3_flat, W5) + b

    ########################################################################
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.batch(100) #배치사이즈
    iterator = dataset.make_initializable_iterator() #함수 호출, 
    x_epoch, y_epoch = iterator.get_next() #next batch
    step = 0
    
    with tf.Session() as sess:
        #haha = tf.summary.FileWriter('/tmp/1', sess.graph)
        sess.run(tf.global_variables_initializer())
        tmplst1 = []
        tmplst2 = []
        a = []    
        for epoch in range(20):
            sess.run(iterator.initializer) #iterator 상태를 처음 초기화하

            while True:
                try:
                    batch_x_data, batch_y_data = sess.run([x_epoch, y_epoch])
                    loss_v, _ = sess.run([cost,optimizer], feed_dict={X:batch_x_data,Y:batch_y_data, keep_prob: 1.0}) # 드롭아웃 일단 뺌
                    loss_x = sess.run(cost, feed_dict={X:X_test,Y:y_test, keep_prob: 1.0})

                    tmplst1.append(loss_v)
                    tmplst2.append(loss_x)

                    step = step +100

                    if step % 100 == 0:
                        print(epoch, '|', step, '|', loss_v, '|', abs(loss_v - loss_x))
                        a.append(abs(loss_v - loss_x))
                                            
                except tf.errors.OutOfRangeError:
                    step = 0
                    break
        acc_v = sess.run(accuracy, feed_dict={X:X_test,Y:y_test, keep_prob: 1.0})
        print(tf.argmax(a))
        print(acc_v)

        plt.plot(range(len(tmplst1)), tmplst1)
        plt.plot(range(len(tmplst2)), tmplst2, c ='r')
        plt.show()
        

add1 = "cosmetics/NARS"
add2 = "cosmetics/YSL"

model(add1, add2)


# ### Early stopping(코스트 minimum 초과 시 종료)

# In[ ]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


get_ipython().run_line_magic('matplotlib', 'inline')


def model(NARS, YSL, patience):
    NARS_data = os.listdir(NARS)
    print(len(NARS_data))
    YSL_data = os.listdir(YSL)
    print(len(YSL_data))

    c=[]
    for i in range(len(NARS_data)):
        a=cv2.imread('./cosmetics/NARS/{}'.format(NARS_data[i]))
        a=cv2.resize(a, dsize=(50, 50))
        c.append(a)

    for i in range(len(YSL_data)):
        b=cv2.imread('./cosmetics/YSL/{}'.format(YSL_data[i]))
        b=cv2.resize(b, dsize=(50, 50))
        c.append(b)

    label=[]
    for i in range(len(NARS_data)):
        label.append(1)


    for i in range(len(YSL_data)):
        label.append(0)
  
    X_train, X_test, y_train, y_test = train_test_split(c, label, test_size=0.75, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    X_train = np.array(X_train)
#     X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
#     y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    from sklearn.preprocessing import OneHotEncoder

    y_train=y_train.reshape([-1,1])
    enc=OneHotEncoder()
    enc.fit(y_train)
    y_train=enc.transform(y_train).toarray()
    
#     y_val = y_val.reshape([-1,1])
#     enc = OneHotEncoder()
#     enc.fit(y_val)
#     y_val = enc.transform(y_val).toarray()
    
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

    X = tf.placeholder(tf.float32, shape=[None, 50,50,3])  
    Y = tf.placeholder(tf.float32, [None, 2])

    #50*50*3
    W1 = tf.Variable(tf.random_normal([5, 5, 3, 16], stddev=0.01))
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    #25*25*16
    W2 = tf.Variable(tf.random_normal([5, 5, 16, 32], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    #13*13*32

    W3 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    W4 = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev=0.01))
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    L3_flat = tf.reshape(L4, [-1, 4 * 4 * 128])
    #7*7*64

    W5 = tf.Variable(tf.random_normal([4 * 4 * 128, 2],stddev=0.01))
    b = tf.Variable(tf.random_normal([2]))
    logits = tf.matmul(L3_flat, W5) + b

    ########################################################################
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.batch(100) #배치사이즈
    iterator = dataset.make_initializable_iterator() #함수 호출, 
    x_epoch, y_epoch = iterator.get_next() #next batch
    loss = 0
    count = 0
    totalBreak = True
    
    with tf.Session() as sess:
        #haha = tf.summary.FileWriter('/tmp/1', sess.graph)
        sess.run(tf.global_variables_initializer())
        tmplst1 = []
        tmplst2 = []
        epoch = 0
        prv = None
#         for epoch in range(60):
        while True:
            epoch += 1
            sess.run(iterator.initializer) #iterator 상태를 처음 초기화하

            while True:
                try:
                    batch_x_data, batch_y_data = sess.run([x_epoch, y_epoch])
                    loss, _ = sess.run([cost,optimizer], feed_dict={X:batch_x_data,Y:batch_y_data, keep_prob: 1.0}) # 드롭아웃 일단 뺌
                                        
                except tf.errors.OutOfRangeError:
                    step = 0
                    break
                    
                if totalBreak == False:
                    break
            
            loss_t = sess.run(cost, feed_dict={X:batch_x_data, Y:batch_y_data, keep_prob: 1.0})
            loss_x = sess.run(cost, feed_dict={X:X_test,Y:y_test, keep_prob: 1.0})
                        
            tmplst1.append(loss_t)
            tmplst2.append(loss_x)
            
            if epoch == 1 or loss_x < prv:
                prv = loss_x
                count = 0
#                 savemodel()
            else:
                count += 1
            
            print(epoch, '| loss :', loss_x, 'count :', count)

            if count >= patience:
                print(count)
                break
                
        acc_v = sess.run(accuracy, feed_dict={X:X_test,Y:y_test, keep_prob: 1.0})
        
        print(tf.argmax(a))
        print(acc_v)

        plt.plot(range(len(tmplst1)), tmplst1)
        plt.plot(range(len(tmplst2)), tmplst2, c ='r')
        plt.show()
        

add1 = "cosmetics/NARS"
add2 = "cosmetics/YSL"

model(add1, add2, 4)


# ### 2중 반복문 연습

# In[ ]:


num = 0
breakIf = True
for i in range(10):
    while True:
        num = num + 1
        if num == 8:
            breakIf = False
            break
        print(num)
    
    if breakIf == False:
        break
        


# ### Early stopping(train과 test의 차의 절대값을 이용)

# In[ ]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


get_ipython().run_line_magic('matplotlib', 'inline')


def model(NARS, YSL, patience):
    NARS_data = os.listdir(NARS)
    print(len(NARS_data))
    YSL_data = os.listdir(YSL)
    print(len(YSL_data))

    c=[]
    for i in range(len(NARS_data)):
        a=cv2.imread('./cosmetics/NARS/{}'.format(NARS_data[i]))
        a=cv2.resize(a, dsize=(50, 50))
        c.append(a)

    for i in range(len(YSL_data)):
        b=cv2.imread('./cosmetics/YSL/{}'.format(YSL_data[i]))
        b=cv2.resize(b, dsize=(50, 50))
        c.append(b)

    label=[]
    for i in range(len(NARS_data)):
        label.append(1)


    for i in range(len(YSL_data)):
        label.append(0)

    '''
    NARS_tr=c[:len(NARS_data)-100]#훈련시킬 아반떼 o
    NARS_te=c[len(NARS_data)-100:] #학습시킬 아반떼 o
    YSL_tr=d[:len(YSL_data)-100]#훈련시킬 아반떼 x
    YSL_te=d[len(YSL_data)-100:] #학습시킬 아반떼 x

    NARS_l=NARS_label[:len(NARS_label)-100]
    NARS_tl=NARS_label[len(NARS_label)-100:]
    YSL_l=YSL_label[:len(YSL_label)-100]
    YSL_tl=YSL_label[len(YSL_label)-100:]
    NARS_train=NARS_tr+YSL_tr #학습 이미지
    NARS_test=NARS_te+YSL_te #테스트 이미지
    YSL_train=NARS_l+YSL_l #학습 라벨
    YSL_test=NARS_tl+YSL_tl#테스트 라벨

    NARS_train=np.array(NARS_train)
    NARS_test=np.array(NARS_test)
    YSL_train=np.array(YSL_train)
    YSL_test=np.array(YSL_test)
    '''

    X_train, X_test, y_train, y_test = train_test_split(c, label, test_size=0.33, random_state=42)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    from sklearn.preprocessing import OneHotEncoder

    y_train=y_train.reshape([-1,1])
    enc=OneHotEncoder()
    enc.fit(y_train)
    y_train=enc.transform(y_train).toarray()
    
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

    X = tf.placeholder(tf.float32, shape=[None, 50,50,3])  
    Y = tf.placeholder(tf.float32, [None, 2])

    #50*50*3
    W1 = tf.Variable(tf.random_normal([5, 5, 3, 16]))
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    #25*25*16
    W2 = tf.Variable(tf.random_normal([5, 5, 16, 32]))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    #13*13*32

    W3 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    W4 = tf.Variable(tf.random_normal([5, 5, 64, 128]))
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    L3_flat = tf.reshape(L4, [-1, 4 * 4 * 128])
    #7*7*64

    W5 = tf.Variable(tf.random_normal([4 * 4 * 128, 2]))
    b = tf.Variable(tf.random_normal([2]))
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
    
    with tf.Session() as sess:
        #haha = tf.summary.FileWriter('/tmp/1', sess.graph)
        sess.run(tf.global_variables_initializer())
        tmplst1 = []
        tmplst2 = []
        a = []    
        absSub = 0
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
                        
            loss_x = sess.run(cost, feed_dict = {X:X_test, Y:y_test, keep_prob: 1.0})
            
            if epoch > 1 and absSub < abs(loss_v - loss_x):
                count += 1
            
            print('epoch :', epoch, '|', 'loss_v :', loss_v, '|', 'abs(avg_loss - loss_x) :', abs(loss_v - loss_x), '|', 'count :', count)
            a.append(abs(loss_v - loss_x))
            
            tmplst1.append(loss_v)
            tmplst2.append(loss_x)
            
            if count == patience:
                break
            
            absSub = abs(loss_v - loss_x)
            
        acc_v = sess.run(accuracy, feed_dict={X:X_test,Y:y_test, keep_prob: 1.0})
        print(tf.argmax(a))
        print(acc_v)

        plt.plot(range(len(tmplst1)), tmplst1)
        plt.plot(range(len(tmplst2)), tmplst2, c ='r')
        plt.show()
        

add1 = "cosmetics/NARS"
add2 = "cosmetics/YSL"

model(add1, add2, 5)


# ### accuracy max값 적용, 데이터 set 3개로 나눔

# In[ ]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


get_ipython().run_line_magic('matplotlib', 'inline')


def model(NARS, YSL, patience):
    NARS_data = os.listdir(NARS)
    print(len(NARS_data))
    YSL_data = os.listdir(YSL)
    print(len(YSL_data))

    c=[]
    for i in range(len(NARS_data)):
        a=cv2.imread('./cosmetics/NARS/{}'.format(NARS_data[i]))
        a=cv2.resize(a, dsize=(50, 50))
        c.append(a)

    for i in range(len(YSL_data)):
        b=cv2.imread('./cosmetics/YSL/{}'.format(YSL_data[i]))
        b=cv2.resize(b, dsize=(50, 50))
        c.append(b)

    label=[]
    for i in range(len(NARS_data)):
        label.append(1)


    for i in range(len(YSL_data)):
        label.append(0)

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

    X = tf.placeholder(tf.float32, shape=[None, 50,50,3])  
    Y = tf.placeholder(tf.float32, [None, 2])

    #50*50*3
    W1 = tf.Variable(tf.random_normal([5, 5, 3, 16], stddev=0.01))
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    #25*25*16
    W2 = tf.Variable(tf.random_normal([5, 5, 16, 32], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    #13*13*32

    W3 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    W4 = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev=0.01))
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    L3_flat = tf.reshape(L4, [-1, 4 * 4 * 128])
    #7*7*64

    W5 = tf.Variable(tf.random_normal([4 * 4 * 128, 2], stddev=0.01))
    b = tf.Variable(tf.random_normal([2]))
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
    
    with tf.Session() as sess:
        #haha = tf.summary.FileWriter('/tmp/1', sess.graph)
        sess.run(tf.global_variables_initializer())
        tmplst1 = []
        tmplst2 = []
        a = []    
        absSub = 0
        acc = 0
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
            
            if epoch > 1 and absSub < abs(loss_v - loss_x):
                count += 1
            
            if epoch > 1 and acc < acc_v:
                acc = acc_v
                count += 2
                        
            print('epoch :', epoch, '|', 'loss_v :', loss_v, '|', 'abs(avg_loss - loss_x) :', abs(loss_v - loss_x), '|', 'count :', count)
            a.append(abs(loss_v - loss_x))
            
            tmplst1.append(loss_v)
            tmplst2.append(loss_x)
            
            if count >= patience * 2:
                break
            
            absSub = abs(loss_v - loss_x)
            
        acc_v = sess.run(accuracy, feed_dict={X:X_val,Y:y_val, keep_prob: 1.0})
        print(tf.argmax(a))
        print('acc_v :', acc_v)
        print('acc :', acc)

        plt.plot(range(len(tmplst1)), tmplst1)
        plt.plot(range(len(tmplst2)), tmplst2, c ='r')
        plt.show()
        

add1 = "cosmetics/NARS"
add2 = "cosmetics/YSL"

model(add1, add2, 5)


# ### valSet loss 최저값 : 4, 데이터 set 3개로 나눔 + 모델 저장

# In[ ]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


get_ipython().run_line_magic('matplotlib', 'inline')


def model(NARS, YSL, patience):

    NARS_data = os.listdir(NARS)
    print(len(NARS_data))
    YSL_data = os.listdir(YSL)
    print(len(YSL_data))

    c=[]
    for i in range(len(NARS_data)):
        a=cv2.imread('./cosmetics/NARS/{}'.format(NARS_data[i]))
        a=cv2.resize(a, dsize=(50, 50))
        c.append(a)

    for i in range(len(YSL_data)):
        b=cv2.imread('./cosmetics/YSL/{}'.format(YSL_data[i]))
        b=cv2.resize(b, dsize=(50, 50))
        c.append(b)

    label=[]
    for i in range(len(NARS_data)):
        label.append(1)


    for i in range(len(YSL_data)):
        label.append(0)

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
    Y = tf.placeholder(tf.float32, [None, 2], name='Y')

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

    W5 = tf.Variable(tf.random_normal([4 * 4 * 128, 2], stddev=0.01), name='W5')
    b = tf.Variable(tf.random_normal([2]))
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
        absSub = 0
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

            if epoch > 1 and absSub < abs(loss_v - loss_x):
                count += 1

            if epoch == 1:
                lossMin = loss_x
            elif epoch > 1 and lossMin > loss_x:
                count += 4
                lossMin = loss_x

            print('epoch :', epoch, '|', 'loss_v :', loss_v, '|', 'abs(avg_loss - loss_x) :', abs(loss_v - loss_x), '|', 'count :', count)
            a.append(abs(loss_v - loss_x))

            tmplst1.append(loss_v)
            tmplst2.append(loss_x)

            if count >= patience * 3:
                break

            absSub = abs(loss_v - loss_x)
            
        saver.save(sess, save_file)
        acc_v = sess.run(accuracy, feed_dict={X:X_val,Y:y_val, keep_prob: 1.0})
        print(tf.argmax(a))
        print('acc_v :', acc_v)

        plt.plot(range(len(tmplst1)), tmplst1)
        plt.plot(range(len(tmplst2)), tmplst2, c ='r')
        plt.show()

add1 = "cosmetics/NARS"
add2 = "cosmetics/YSL"
patience = 5

model(add1, add2, patience)


# ### 위의 모델 불러오기

# In[ ]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

NARS = "cosmetics/NARS"
YSL = "cosmetics/YSL"
patience = 5

NARS_data = os.listdir(NARS)
print(len(NARS_data))
YSL_data = os.listdir(YSL)
print(len(YSL_data))

c=[]
for i in range(len(NARS_data)):
    a=cv2.imread('./cosmetics/NARS/{}'.format(NARS_data[i]))
    a=cv2.resize(a, dsize=(50, 50))
    c.append(a)

for i in range(len(YSL_data)):
    b=cv2.imread('./cosmetics/YSL/{}'.format(YSL_data[i]))
    b=cv2.resize(b, dsize=(50, 50))
    c.append(b)

label=[]
for i in range(len(NARS_data)):
    label.append(1)


for i in range(len(YSL_data)):
    label.append(0)

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
Y = tf.placeholder(tf.float32, [None, 2], name='Y')

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

W5 = tf.Variable(tf.random_normal([4 * 4 * 128, 2], stddev=0.01), name='W5')
b = tf.Variable(tf.random_normal([2]))
logits = tf.matmul(L3_flat, W5) + b

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

save_file = './model/model.ckpt'

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_file)
 
    acc = sess.run(accuracy, feed_dict = {X:X_test, Y:y_test, keep_prob: 1.0})
    
    print(acc)


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
        #haha = tf.summary.FileWriter('/tmp/1', sess.graph)
        sess.run(tf.global_variables_initializer())
        tmplst1 = []
        tmplst2 = []
        a = []    
        absSub = 0
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

            if epoch > 1 and absSub < abs(loss_v - loss_x):
                count += 1

            if epoch == 1:
                lossMin = loss_x
            elif epoch > 1 and lossMin > loss_x:
                count += 4
                lossMin = loss_x

            print('epoch :', epoch, '|', 'loss_v :', loss_v, '|', 'abs(avg_loss - loss_x) :', abs(loss_v - loss_x), '|', 'count :', count)
            a.append(abs(loss_v - loss_x))

            tmplst1.append(loss_v)
            tmplst2.append(loss_x)

            if count >= patience * 3:
                break

            absSub = abs(loss_v - loss_x)
            
        saver.save(sess, save_file)
        acc_v = sess.run(accuracy, feed_dict={X:X_val,Y:y_val, keep_prob: 1.0})
        print(tf.argmax(a))
        print('acc_v :', acc_v)

        plt.plot(range(len(tmplst1)), tmplst1)
        plt.plot(range(len(tmplst2)), tmplst2, c ='r')
        plt.show()
        
add1 = "img/_pre/"
patience = 5

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

