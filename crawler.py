# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from konlpy.tag import Hannanum
from konlpy.tag import Twitter
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from urllib.parse import quote_plus
from urllib.request import urlopen

import konlpy.tag
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import platform
import re
import time

def crawl(sns_id, passwd, keyword, count, site):
	start = time.time()
	'''
	keyword : 검색할 키워드
	count : 스크롤할 횟수
	site == 0 : 인스타
	site == 1 : 핀터
	site == 2 : 페북
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
	driver.implicitly_wait(5) # 최대 5초 대기
	
	if site == 0:
		driver.get("https://www.instagram.com/")
		elem = driver.find_element_by_name('emailOrPhone')
		elem.send_keys(sns_id)
		elem = driver.find_element_by_name('password')
		elem.send_keys(passwd)
		elem.send_keys(Keys.RETURN)
		# 로그인
		time.sleep(3) # TODO 로딩 문제 해결되면 지우기..
		baseUrl = 'https://www.instagram.com/explore/tags/'
		url = baseUrl + quote_plus(keyword)
		# time.sleep(3) # TODO delete
		# WebDriverWait # TODO 가장 이상적인 방식이지만, 수정할 내용이 많음.
		driver.get(url)
		elem = driver.find_element_by_tag_name('body')
		page1 = driver.page_source
		soup1 = BeautifulSoup(page1, 'html.parser')
		wrong_tag = soup1.find_all('div', 'error-container -cx-PRIVATE-ErrorPage__errorContainer -cx-PRIVATE-ErrorPage__errorContainer__')
		
		if len(wrong_tag): # 잘못된 태그 사용
			print("error code {}".format(-1))
			return -1
		
		img_url_list = []
		img_list = []
		box = [1, 2, 3, 4]
		n = 1
		pagedowns = 1
		prv_len_img = None

		while True:
			elem = driver.find_element_by_tag_name('body')
			page1 = driver.page_source
			soup1 = BeautifulSoup(page1, 'html.parser')
			no_tag = soup1.find_all('div', '_4Kbb_ _wTvQ')

			if len(no_tag): # 해당 태그 게시글 없음
				print("error code {}".format(-2))
				return -2

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

			print("len img_url_list :", len(img_url_list))
			print("len img_list :", len(img_list))

			cur_len_img = len(img_list)

			# 로딩이 끝났으면 0,
			# 로딩할게 남았으면 1
			loading = len(soup1.find_all('div', '_4emnV'))

			if prv_len_img is None:
				prv_len_img = cur_len_img
			else:
				if cur_len_img >= count or loading == 0:
					with open("imgurls.txt", 'w') as file:
						for line in img_url_list:
							file.write(line + '\n')
					print(cur_len_img, "image(s)")
					break
				elif cur_len_img < count:
					prv_len_img = cur_len_img
				else: # 예외 발생
					print("error code {}".format(-2))
					raise
			elem.send_keys(Keys.PAGE_DOWN)

		for i, img in enumerate(img_list):
			with urlopen(img) as f:
				if not os.path.isdir(img_dir):
					os.makedirs(os.path.join(img_dir))
				with open(img_dir + keyword + str(i+1) + '.jpg', 'wb') as h:
					img = f.read()
					h.write(img)
	elif site == 1:
		baseUrl = 'https://www.pinterest.co.kr/'
		#ps=input('입력')
		#count = input('스크롤할 횟수를 입력하세요 : ')
		driver=webdriver.Chrome('chromedriver')
		driver.get(baseUrl)

		driver.find_element_by_id('email').send_keys('qweewq1111@naver.com')
		driver.find_element_by_id('password').send_keys('zzzzzzzz11')
		driver.find_element_by_id('age').send_keys('25')
		driver.find_element_by_id('email').send_keys(Keys.RETURN)

		time.sleep(3)
		plusurl = 'search/pins/?q={0}&rs=typed&term_meta[]={0}%7Ctyped'.format(keyword)

		url = baseUrl + (plusurl)
		driver.get(url)


		page = driver.page_source
		soup = BeautifulSoup(page)
		a2 = soup.find_all('div',"Yl- MIw Hb7")

		elem = driver.find_element_by_tag_name("body")
		imgurl = []

		n=1
		pagedowns=1

		while pagedowns < int(count):
			elem.send_keys(Keys.PAGE_DOWN)
			time.sleep(1)
			pagedowns += 1
			page1 = driver.page_source
			soup1 = BeautifulSoup(page1)
			a2 = soup1.find_all('div',"Yl- MIw Hb7")
			for i in range(0, len(a2)):
				imgUrl = a2[i].select_one('.hCL.kVc.L4E.MIw').get('src')
				imgurl.append(imgUrl)
			imgurl = list(set(imgurl))
		save_dir = './out/{}/'.format(time.strftime('%y%m%d_%H%M%S', time.localtime(time.time())))
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
	else:
		raise
		
	driver.quit()
	print("It takes {:.2f} second(s)".format(time.time() - start))

crawl("jonson131214@gmail.com", "q1w2e3r4!@", "나스", 200, 0)