# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from konlpy.tag import Hannanum
from konlpy.tag import Twitter
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from urllib.request import urlopen

import konlpy.tag
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import platform
import re
import time


def crawl(sns_id, passwd, keyword, count=100, site=0):
	start = time.time()
	save_dir = 'crawl/{}_{}/'.format(time.strftime('%y%m%d_%H%M%S', time.localtime(time.time())), keyword)
	img_dir = save_dir + 'img/'
	'''
	keyword : 검색할 키워드
	count : 스크롤할 횟수
	site == 0 : 인스타
	site == 1 : 핀터
	site == 2 : 페북
	'''
	# options = webdriver.ChromeOptions()
	# options.add_argument('headless')
	# options.add_argument('window-size=1920x1080')
	# options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")
	# options.add_argument("disable-gpu") # GPU 가속이 문제가 될 경우 해당 옵션 활성화 하여, GPU 가속 끄기 or
	# options.add_argument("--disable-gpu") # GPU 가속이 문제가 될 경우 해당 옵션 활성화 하여, GPU 가속 끄기
	# driver = webdriver.Chrome('chromedriver', chrome_options=options)
	driver = webdriver.Chrome('chromedriver')
	driver.implicitly_wait(5) # 최대 5초 대기
	
	if site == 0:
		baseUrl = 'https://www.instagram.com/explore/tags/'
		url = baseUrl + keyword
		driver.get(url)
		
		img_url_list = []
		img_list = []

		while True:
			##### HTML 가져오기 #####
			elem = driver.find_element_by_tag_name('body')
			page1 = driver.page_source
			soup1 = BeautifulSoup(page1, 'html.parser')

			##### 잘못된 태그 사용 #####
			wrong_tag = soup1.find_all('div', 'error-container -cx-PRIVATE-ErrorPage__errorContainer -cx-PRIVATE-ErrorPage__errorContainer__')
			
			if len(wrong_tag):
				print("error code {}".format(-1))
				return -1

			##### 해당 태그 게시글 없음 #####
			no_tag = soup1.find_all('div', '_4Kbb_ _wTvQ')

			if len(no_tag):
				print("error code {}".format(-2))
				return -2

			##### 로그인 요청 #####
			need_login = soup1.find_all('div', 'Igw0E IwRSH eGOV_ _4EzTm IM32b')

			if len(need_login):
				driver.find_element_by_name('username').send_keys(sns_id)
				driver.find_element_by_name('password').send_keys(passwd)
				driver.find_element_by_name('password').send_keys(Keys.RETURN)

				##### 로그인 요청 대기 #####
				login_wait = soup1.find_all('button', 'sqdOP L3NKy y3zKF')

				while len(login_wait):
					##### HTML 가져오기 #####
					elem = driver.find_element_by_tag_name('body')
					page1 = driver.page_source
					soup1 = BeautifulSoup(page1, 'html.parser')
					##### 종료 조건 확인 #####
					login_wait = soup1.find_all('button', 'sqdOP L3NKy y3zKF')

				##### 로그인 실패 #####
				login_fail = soup1.find_all('div', 'eiCW')
				
				if len(login_fail):
					return -34

				continue

			##### 이미지 링크 존재 유무 확인 #####
			img_urls = soup1.find_all('div', 'v1Nh3 kIKUG _bz0w')

			if len(img_urls) == 0:
				continue

			images = soup1.find_all('div', 'KL4Bh')

			for item in img_urls:
				url_text = 'https://www.instagram.com' + item.find('a')["href"]
				
				if url_text not in img_url_list:
					img_url_list.append(url_text)

			for item in images:
				try:
					img = item.find('img')['src']
				except KeyError:
					print("error code {}".format(-3))

				if img not in img_list:
					img_list.append(img)

			cur_len_img = len(img_list)

			# 로딩이 끝났으면 0,
			# 로딩할게 남았으면 1
			loading = len(soup1.find_all('div', '_4emnV'))

			##### 종료조건 #####
			if cur_len_img >= count or loading == 0:
				print("cur_len_img :", cur_len_img)
				print("count :", count)
				print("loading :", loading)

				if not os.path.isdir(img_dir):
					os.makedirs(os.path.join(img_dir))
				print(cur_len_img, "image(s)")

				break

			elem.send_keys(Keys.PAGE_DOWN)

		for i, img in enumerate(img_list):
			filename = keyword + str(i+1) + '.jpg'

			try:
				with urlopen(img) as f:
					with open(img_dir + filename, 'wb') as h:
						img = f.read()
						h.write(img)
				with open("{}url.csv".format(save_dir), 'a') as file:
					file.write(img_url_list[i] + ',')
					file.write(filename + '\n')
			except TimeoutError:
				print("err - 127")
				
	elif site == 1:
		baseUrl = 'https://www.pinterest.co.kr/'
		loginUrl = baseUrl + 'login/'
		driver.get(loginUrl)

		driver.find_element_by_id('email').send_keys(sns_id)
		driver.find_element_by_id('password').send_keys(passwd)
		driver.find_element_by_id('password').send_keys(Keys.RETURN)

		time.sleep(1) # TODO 로딩 문제 해결되면 지우기..

		plusurl = 'search/pins/?q={0}'.format(keyword)

		url = baseUrl + plusurl
		print(url)
		driver.get(url)

		page = driver.page_source
		soup = BeautifulSoup(page)
		a2 = soup.find_all('div',"Yl- MIw Hb7")

		elem = driver.find_element_by_tag_name("body")
		imgurl = []

		n=1
		pagedowns=1

		while pagedowns < int(count):
			time.sleep(1)
			elem.send_keys(Keys.PAGE_DOWN)
			pagedowns += 1
			page1 = driver.page_source
			soup1 = BeautifulSoup(page1)
			a2 = soup1.find_all('div',"Yl- MIw Hb7")
			for i in range(0, len(a2)):
				try:
					imgUrl = a2[i].select_one('.hCL.kVc.L4E.MIw').get('src')
				except AttributeError:
					print("error code {}".format(-5))
				imgurl.append(imgUrl)
			imgurl = list(set(imgurl))

		save_dir = './crawl/{}/'.format(time.strftime('%y%m%d_%H%M%S', time.localtime(time.time())))
		img_dir = save_dir + 'img/'
		n=1
		for i in imgurl:
			with urlopen(i) as f:
				if not os.path.isdir(img_dir):
					os.makedirs(os.path.join(img_dir))
				with open(img_dir + keyword + str(n) + '.jpg', 'wb') as h:
					img = f.read()
					h.write(img)
					n+=1
		print(len(imgurl), "image(s)")
	else:
		raise
		
	driver.quit()
	print("It takes {:.2f} second(s)".format(time.time() - start))