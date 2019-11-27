def analyse():

			# if prv_len_img is None:
			# 	prv_len_img = cur_len_img
			# else:
			# 	if cur_len_img >= count or loading == 0:
			# 		with open("imgurls.txt", 'w') as file:
			# 			for line in img_url_list:
			# 				file.write(line + '\n')
			# 		print(cur_len_img, "image(s)")
			# 		break
			# 	elif cur_len_img < count:
			# 		prv_len_img = cur_len_img
			# 	else: # 예외 발생
			# 		print("error code {}".format(-2))
			# 		raise

		# main_text = []
		# comment = []
		# sub_comment = []

		# for i, url in enumerate(img_url_list):
		# 	driver.get(url)
		# 	html = driver.page_source
		# 	soup = BeautifulSoup(html, 'html.parser')
		# 	insta = soup.select('._97aPb.wKWK0')
		# 	for j in insta:
		# 		while True:
		# 			try:
		# 				imgUrl = j.select_one('.KL4Bh').img['src']
		# 				with urlopen(imgUrl) as f:
		# 					if not os.path.isdir(img_dir):
		# 						os.makedirs(os.path.join(img_dir))
		# 					with open(img_dir + keyword + str(i+1) + '.jpg', 'wb') as h:
		# 						img = f.read()
		# 						h.write(img)
		# 				break
		# 			except AttributeError:
		# 				print(url, i, '번째 이미지는 동영상입니다.')
		# 				# try:
		# 				# 	imgUrl = j.select_one('._5wCQW').img['src']
		# 				# 	with urlopen(imgUrl) as f:
		# 				# 		if not os.path.isdir(img_dir):
		# 				# 			os.makedirs(os.path.join(img_dir))
		# 				# 		with open(img_dir + keyword + str(i+1) + '.jpg', 'wb') as h:
		# 				# 			img = f.read()
		# 				# 			h.write(img)
		# 				# 	break
		# 				# except TypeError:
		# 				# 	print(url, i, '번째 이미지가 없습니다.')
		# 				break	
		soup = BeautifulSoup(html, 'lxml')
		insta = soup.select('.C4VMK')
		
		while True:
			if insta is not None:
				for j in range(0, 1):
					try:
						title = [insta[j].find('span').text]
					except IndexError:
						print("erroe_code:127")
						break
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