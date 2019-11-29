import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import re
import time
import os
import konlpy.tag
from bs4 import BeautifulSoup
from konlpy.tag import Hannanum
from konlpy.tag import Twitter
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from urllib.parse import quote_plus
from urllib.request import urlopen
import konlpy.tag

#문장단위 데이터를 형태소 기준 태깅 해서 Json 파일로 만들기 위함임.
import json
import numpy as np; np.random.seed(1234)
from pprint import pprint

#konlpy 한글 정보처리를 위한 파이썬 패키지 (형태소 분석기로 사용)
import konlpy
from konlpy.tag import Okt

#그래프를 만들어주기 위함.
import nltk
from matplotlib import font_manager, rc

#머신러닝을 위한 텐서플로우.케라스를 사용하기위해 임포트 시킴.
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

#케라스 머신러닝 모델을 저장하고 다시 불러오기 위함.
from keras.models import load_model
def text(file_dir):
    files = os.listdir(file_dir)

    anaList = []
    for file in files:
        if file[-3:] == "jpg":
            anaList.append(file)

    file_dir = file_dir.split('/')[-2]
    keyword = file_dir.split('_')[-1]

    save_dir = './crawl/'
    url_dir = save_dir + file_dir + '/url.csv'
    df = pd.read_csv(url_dir, encoding = 'euc-kr', header = None)

    urlList = []
    for ana in anaList:
        url = df[df[0] == ana][1].tolist()[0]
        urlList.append(url)

    save_dir = './ana/'
    full_path = save_dir+file_dir + '/'

    if not os.path.isdir(full_path):
        os.makedirs(full_path)
    with open(full_path + "/url.csv", 'a') as f:
        for url in urlList:
            f.write(url + '\n')

    df = pd.from_from_csv(full_path + "/url.csv")

    imgurl = []
    for i in range(len(df[0])):
        url = df[0][i]
        imgurl.append(url)

    driver = webdriver.Chrome('chromedriver')
    driver.implicitly_wait(2)
    sns_id = 'jonson131214@gmail.com'
    passwd = 'q1w2e3r4!@'
    driver.get("https://www.instagram.com/")
    elem = driver.find_element_by_name('emailOrPhone')
    elem.send_keys(sns_id)
    elem = driver.find_element_by_name('password')
    elem.send_keys(passwd)
    elem.send_keys(Keys.RETURN)
    # 로그인
    time.sleep(4)

    main_text = []
    comment = []
    sub_comment = []
    for i in range(len(imgurl)):
        url = imgurl[i]
        driver.get(url)

        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        insta = soup.select('.C4VMK')

        while True:
            if insta is not None:
                for j in range(0, 1):
                        title = [insta[j].find('span').text]
                        sub_comment = []
                        com = ' '
                        main_text.append(title)
                for k in range(1, len(insta)):
                    com = insta[k].find('span').text
                    sub_comment.append(com)
                comment.append(sub_comment)
                break

    uurl = []

    for i in range(0, len(imgurl)):
        url = imgurl[i]
        uurl.append(url)
    data = {'주소' : uurl, '본문' : main_text, '댓글' : comment}

    df = pd.DataFrame(data)
    try:
        df.to_excel(save_dir+file_dir+'/{}.xlsx'.format(keyword), index = False,
                sheet_name = 'sheet1')
    except:
        print(keyword)

    hangul = re.compile('[^ ㄱ-ㅣ가-힣a-zA-Z0-9]+')
    c = []
    d = []
    for i in range(len(main_text)):
        main_text[i][0] = main_text[i][0].replace('#', ' ')
        result = hangul.sub('', main_text[i][0])
        c.append(result)

    data = {'본문' : c}
    df = pd.DataFrame(data)
    df.to_excel(save_dir+file_dir+'/{}Modified.xlsx'.format(keyword),
                index = False, sheet_name = 'sheet1')
    cnt  = 1
    #여기서 부터 감성분석을 합치는 중입니다.

    #형태소 분석하기 위한 라이브러리입니다.
    okt = Okt()

    #1 이부분은 머신에 긍정 부정을 학습을 시키기 위한 과정. 학습시킬 문장 텍스트를 
    #불러와서 전처리 하는 과정입니다.
    def tokenize(doc):
        # norm은 정규화, stem은 근어로 표시하기를 나타냄
        return ['/'.join(t) for t in okt.pos(doc, norm = True, stem = True)]
    #1.1파일을 읽어와서 라인 by 라인으로 세팅하는 함수.
    def read_data(filename):
        #encoding 을 UTF8로 해야, r을 rt로 바꿔야 에러가 안났다.
        with open(filename, 'rt', encoding = 'UTF8') as f:
            data = [line.split('\t') for line in f.read().splitlines()]
            data = data[1:]
        return data
    #1.2ratings_train.txt, ratings_test.txt 는 네이버 영화리뷰를 긁어온 파일임.
    #1.3이를 read_data()함수로 읽어옴  = > 라인 by 라인으로 세팅한다는 뜻

    train_data = read_data('./ratings_train.txt')
    test_data = read_data('./ratings_test.txt')

    if os.path.isfile('./train_docs.json'):
        with open('./train_docs.json', encoding =  "utf-8") as f:
            train_docs = json.load(f)
        with open('./test_docs.json', encoding =  "utf-8") as f:
            test_docs = json.load(f)

    #1.8 태깅된 파일이 없으면, 새로 태깅 시켜 파일로 저장합니다.       
    else:
    #1.9 train_docs에는 tokenize()함수를 써서 train_data의 한줄씩 
    #형태소 분석(태깅)을 해 넣습니다.
        train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    #1.10 test_docs에는 tokenize()함수를 써서 test_data의 한줄씩 
    #형태소 분석(태깅)을 해 넣습니다.
        test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
    #1.11 태깅된 train_docs, test_docs을 .json 파일로 각각 저장합니다. 
        with open('./train_docs.json', 'w', encoding =  "utf-8") as make_file:
            json.dump(train_docs, make_file, ensure_ascii = False, indent = "\t")
        with open('./test_docs.json', 'w', encoding = "utf-8") as make_file:
            json.dump(test_docs, make_file, ensure_ascii = False, indent = "\t")

    #1.12 태깅한 json 파일에서 token들 만을 떼온다.(최종 토큰화 작업)
    tokens = [t for d in train_docs for t in d[0]]
    text = nltk.Text(tokens, name = 'NMSC')
    selected_words = [f[0] for f in text.vocab().most_common(100)]

    #1.21 (중요)학습된 문장내에서 가장 많이 언급된 (현재)100개의 토큰을
    #기준으로 들어온 문장을 감성분석해줍니다. 
    def term_frequency(doc):
        return [doc.count(word) for word in selected_words]
    #2 여기서부터는 전처리된 문장(텍스트)들을 학습시키는 과정입니다. 
    #2.1 model.h5 라는 훈련된 파일이 있으면, 그걸 읽어옵니다. 
    if os.path.isfile('model.h5'):
        model = tf.keras.models.load_model('model.h5')
        #2.2 (중요) 자주사용되는 토큰 100개를 선택함(2019.11.22).
        #원래는 10000개(바꿀려면 다른것들도 다 맞춰야함.)
        selected_words = [f[0] for f in text.vocab().most_common(100)]

    #2.3 model.h5 파일이 없으면, 새로 학습시켜 파일로 저장합니다.       
    else:
        #2.4 자주사용되는 토큰 100개를 선택함. 원래는 10000개
        selected_words = [f[0] for f in text.vocab().most_common(100)]

        #2.5 train_docs에는 토큰화된 훈련 데이터가 있다. 
        #2.6 train_x에 train_docs를 tern_frequency()함수를 써 벡터화한다.
        train_x = [term_frequency(d) for d, _ in train_docs]
        test_x = [term_frequency(d) for d, _ in test_docs]

        train_y = [c for _, c in train_docs]
        test_y = [c for _, c in test_docs]

        #2.7 데이터들을 Float 형으로 변환시켜준다. 여기까지가 데이터 전처리 과정 끝.

        x_train = np.asarray(train_x).astype('float32')
        x_test = np.asarray(test_x).astype('float32')

        y_train = np.asarray(train_y).astype('float32')
        y_test = np.asarray(test_y).astype('float32')

        model = models.Sequential()\
        #2.9 (중요)원레 input_shape 값은 10000개. 지금 설정은 100개(2019.11.21)
        model.add(layers.Dense(64, activation = 'relu', input_shape = (100,)))
        model.add(layers.Dense(64, activation = 'relu'))
        model.add(layers.Dense(1, activation = 'sigmoid'))

        model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
                     loss = losses.binary_crossentropy,
                     metrics = [metrics.binary_accuracy])

        '''    
        #2.10 (중요)원래 batch_size 값은 512. 지금 설정은 10개(2019.11.21)
        원래 epoch 값은 10. 지금 설정은 3개(2019.11.22)
        '''
        model.fit(x_train, y_train, epochs = 3, batch_size = 10)
        results = model.evaluate(x_test, y_test)

        #print(results)

        #2.11 모델 저장
        model.save('model.h5')
    def predict_pos_neg(review):
        #3.1 새로운 문장을 토큰화
        b = []
        c = []
        d = []
        cc = 0
        dd = 0

        token = tokenize(review)
        #3.2 토큰화된 문장을 비교해줌.
        tf = term_frequency(token)
        #3.3 벡터화
        data = np.expand_dims(np.asarray(tf).astype('float32'), axis = 0)
        #3.4 긍/부정 점수계산
        score = float(model.predict(data))

        if(score > 0.5):
            c = review 
            ac = '{:.2f}% 확률로 긍정'.format(score * 100)
            b.append(c)
            cc +=  1
            return c, ac

        else:
            d =  review 
            ad = '{:.2f}% 확률로 부정'.format((1 - score) * 100)
            b.append(d)
            dd +=  1
            return d, ad

        #data = {'머신러닝 짱짱 수듄 분석결과' : b}

        #df = pd.DataFrame(b)
        #df.to_excel(excel_dir + '{}Sentmental_analysis_counter'.format(keyword) + str(cnt) + '.xlsx', index = False,
        #            sheet_name = 'sheet1')
    df = pd.read_excel(save_dir+file_dir+'/' + '{}Modified.xlsx'.format(keyword))
    c = []
    d = []
    for i in range(len(df['본문'])):
        a = df['본문'][i]
        a = a.replace('[', '')
        a = a.replace(']', '')
        a = a.replace("'", '')
        b = predict_pos_neg(a)
        c.append(b[0])
        d.append(b[1])
    data = {'본문' : c, '분석' : d}
    df = pd.DataFrame(data)
    df.to_excel(save_dir+file_dir+'/' + '{}Sentmental_analysis_counter'.format(keyword)
                + '.xlsx', index = False, sheet_name = 'sheet1')
