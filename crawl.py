# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import re
import time
import os
import konlpy.tag

from urllib.request import urlopen
from urllib.parse import quote_plus
from konlpy.tag import Twitter
from konlpy.tag import Hannanum
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup


def crawl(keyword, count, site):
    '''
    keyword : 검색할 키워드
    count : 스크롤할 횟수
    site == 0 : 인스타
    site == 1 : 페북(구현중)
    '''
    
    save_dir = './out/{}/'.format(time.strftime('%y%m%d_%H%M%S', time.localtime(time.time())))
    img_dir = save_dir + 'img/'
    excel_dir = save_dir + 'excel/'
    
    # options = webdriver.ChromeOptions()
    # options.add_argument('headless')
    # options.add_argument('window-size=1920x1080')
    # options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")
    # options.add_argument("disable-gpu") # GPU 가속이 문제가 될 경우 해당 옵션 활성화 하여, GPU 가속 끄기 or
    # options.add_argument("--disable-gpu") # GPU 가속이 문제가 될 경우 해당 옵션 활성화 하여, GPU 가속 끄기
    # driver = webdriver.Chrome('chromedriver', chrome_options=options)
    driver = webdriver.Chrome('chromedriver')

    if site == 0:
        baseUrl = 'https://www.instagram.com/explore/tags/'
        url = baseUrl + quote_plus(keyword)
        driver.implicitly_wait(1)
        # time.sleep(3) # TODO delete
        # WebDriverWait # TODO 가장 이상적인 방식이지만, 수정할 내용이 많음.
        driver.get(url)
        
        elem = driver.find_element_by_tag_name('body')
        imgurl = []
        box = [1, 2, 3, 4]
        n = 1
        pagedowns = 1

        while pagedowns < int(count):
            elem.send_keys(Keys.PAGE_DOWN)
            time.sleep(1)
            pagedowns += 1
            page1 = driver.page_source
            soup1 = BeautifulSoup(page1, 'html.parser')
            a2 = soup1.find_all('div', 'Nnq7C weEfm')
            for i in range(0, len(a2)):
                for j in range(2):
                    imgurl.append('https://www.instagram.com' + a2[i].find_all('a')[j]['href'])

            imgurl = list(set(imgurl))

        main_text = []
        comment = []
        sub_comment = []

        for i in range(len(imgurl)):
            url = imgurl[i]

            driver.implicitly_wait(3)
            driver.get(url)

            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            insta = soup.select('._97aPb.wKWK0')
            for j in insta:
                while True:
                    try:
                        imgUrl = j.select_one('.KL4Bh').img['src']
                        with urlopen(imgUrl) as f:
                            if not os.path.isdir(img_dir):
                                os.makedirs(os.path.join(img_dir))
                            with open(img_dir + keyword + str(i)
                                    + '.jpg', 'wb') as h:
                                img = f.read()
                                h.write(img)
                        break
                    except AttributeError:
                        try:
                            imgUrl = j.select_one('._5wCQW').img['src']
                            with urlopen(imgUrl) as f:
                                if not os.path.isdir(img_dir):
                                    os.makedirs(os.path.join(img_dir))
                                with open(img_dir + keyword + str(i)
                                        + '.jpg', 'wb') as h:
                                    img = f.read()
                                    h.write(img)
                            break
                        except TypeError:
                            print(url,i,'번째 이미지가 없습니다.')
                            break

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
        data={'주소' : uurl, '본문' : main_text, '댓글' : comment}

        if not os.path.isdir(excel_dir):
            os.makedirs(os.path.join(excel_dir))

        df = pd.DataFrame(data)
        df.to_excel(excel_dir + '{}.xlsx'.format(keyword), index=False,
                    sheet_name='sheet1')
        df = pd.read_excel(excel_dir + '{}.xlsx'.format(keyword))

        aaa = pd.read_excel(excel_dir + '{}.xlsx'.format(keyword),
                            header=None)
        content = aaa[1]
        lines = content

        hanna = Hannanum()
        temp = []
        for i in range(1, len(lines)):
            temp.append(hanna.nouns(lines[i]))

        def flatten(l):
            flatList = []
            for elem in l:
                if type(elem) == list:
                    for e in elem:
                        flatList.append(e)
                else:
                    flatList.append(elem)
            return flatList

        word_list = flatten(temp)
        word_list = pd.Series([x for x in word_list if len(x) > 1])
        d = word_list.value_counts()
        d.to_excel(excel_dir + '{}_main.xlsx'.format(keyword))

        aaa = pd.read_excel(excel_dir + '{}.xlsx'.format(keyword),
                            header=None)
        content = aaa[2]
        lines = content

        temp = []
        for i in range(1, len(lines)):
            temp.append(hanna.nouns(lines[i]))

        word_list = flatten(temp)
        word_list = pd.Series([x for x in word_list if len(x) > 1])

        d = word_list.value_counts()
        d.to_excel(excel_dir + '{}_comment.xlsx'.format(keyword))
    elif site == 1:
        pass
    else:
        raise
        
    driver.quit()
