import sys
import PyQt5
import src.preprocessing as pre
import src.crawler as cr
import os
import subprocess

from yolo import YOLO
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets


class MyWindow(QMainWindow):
	def __init__(self, csvFileList, detect_img, ana):
		super().__init__()
		self.crawlDict = {'인스타그램':0, '핀터레스트':0}
		self.detect_img = detect_img
		self.ana = ana

		self.instavalue = 0
		self.pintervalue = 0

		self.setWindowTitle("키워드에 따른 브랜드 통계")
		self.setMinimumSize(QtCore.QSize(410, 210))
		self.setMaximumHeight(200)

		self.centralwidget = QtWidgets.QWidget(self)
		self.centralwidget.setObjectName("centralwidget")

		self.lineEdit = QLineEdit("keyword", self)
		self.lineEdit.move(20, 20)
		self.lineEdit.resize(130, 30)
		self.lineEdit.returnPressed.connect(self.actionstart)

		self.count = QLineEdit("300", self)
		self.count.move(160, 20)
		self.count.resize(30, 30)
		self.count.returnPressed.connect(self.actionstart)

		self.btn1 = QPushButton("크롤링", self)
		self.btn1.move(200, 20)
		self.btn1.resize(70, 30)
		self.btn1.clicked.connect(self.actionstart)

		self.btn2 = QPushButton("검출", self)
		self.btn2.move(280, 20)
		self.btn2.resize(50, 30)
		self.btn2.clicked.connect(self.detect)

		self.btn3 = QPushButton("분석", self)
		self.btn3.move(340, 20)
		self.btn3.resize(50, 30)
		self.btn3.clicked.connect(self.analyse)

		self.trainedBrand = QLabel("- 학습된 브랜드 리스트 -", self)
		self.trainedBrand.move(35, 60)
		self.trainedBrand.resize(150, 30)

		self.siteList = []
		for i, site in enumerate(self.crawlDict.keys()):
			checkBox = QCheckBox(site, self)
			checkBox.move(220 + i * 90, 60)
			checkBox.resize(100, 30)
			checkBox.stateChanged.connect(self.checkBoxState)
			self.siteList.append(checkBox)

		self.siteList[0].setChecked(True)

		self.labelList = []
		for i, csvFile in enumerate(csvFileList):
			filename = pre.getFilenameWoExt(csvFile)

			label = QLabel(filename, self)
			label.move(25 + i // 4 * 100, 90 + i % 4 * 25)
			self.labelList.append(label)

	def checkBoxState(self):
		for i, site in enumerate(self.siteList):
			if site.isChecked() == True:
				self.crawlDict[site.text()] = 1
			else :
				self.crawlDict[site.text()] = 0

	def actionstart(self) :
		keyword = self.lineEdit.text().strip().replace(' ', '')

		if 1 not in self.crawlDict.values():
			QMessageBox.warning(self, "Warning", "크롤링 할 SNS 사이트를 하나 이상 선택해 주세요.")
		else:
			save_dir = os.getcwd() + "\\crawl"
			if not os.path.isdir(save_dir):
				os.makedirs(os.path.join(save_dir))
			# subprocess.call("explorer {}".format(save_dir), shell=True)
			for i, sns in enumerate(self.crawlDict.values()):
				if sns == 1 :
					count = int(self.count.text())
					if i == 0:
						cr.crawl("jonson131214@gmail.com", "q1w2e3r4!@", keyword, count, 0)
					elif i == 1:
						cr.crawl("qweewq1111@naver.com", "zzzzzzzz11", keyword, count, 1)

	def detect(self) :
		path = "./crawl/"
		folderList = os.listdir(path)
		detectList = []

		for folder in folderList:
			if os.path.isfile(path + folder + "/url.csv"):
				detectList.append(path + folder + "/")

		self.newWindow = DetectWindow(detectList, self.detect_img)
		self.newWindow.show()

	def analyse(self) :
		path = "./detect/"
		folderList = os.listdir(path)
		anaList = []

		for folder in folderList:
			anaList.append(path + folder + "/")

		self.newWindow = AnaWindow(anaList, self.ana)
		self.newWindow.show()

	def auto(self):
		searchList = ['샤넬', '닥터자르트', '헤라', '이니스프리', '라네즈', '나스', '설화수', '숨37', '후', '입생로랑', 	'립스틱', '블러셔', '비비', '선크림', '세럼', '아이브로우', '에센스', '마스크', '마스크팩', '카드지갑', '컨실러', '쿠션', '크림', '비비크림', '클러치', '파운데이션']
		for search in searchList:
			self.lineEdit.setText(search)
			self.actionstart()

class DetectWindow(QMainWindow):
	def __init__(self, detectList, detect_img):
		super().__init__()
		self.detect_img = detect_img

		width = 270 + len(detectList) // 10 * 280

		self.setMinimumSize(QtCore.QSize(width, 420))
		self.setMaximumHeight(420)

		detectList.reverse()

		for i, detFolder in enumerate(detectList):
			name = detFolder.split('/')[-2]
			size = 230

			self.btn_detect = QPushButton(name, self)
			self.btn_detect.move(20 + i // 10 * (size + 50), 20 + i % 10 * 40)
			self.btn_detect.resize(size, 30)
			self.btn_detect.setObjectName(detFolder)
			self.btn_detect.clicked.connect(self.detect)

	def detect(self):
		sending_button = self.sender()
		detFolder = str(sending_button.objectName())
		self.detect_img(YOLO(), detFolder + '/img/')


class AnaWindow(QMainWindow):
	def __init__(self, anaList, ana):
		super().__init__()
		self.ana = ana

		width = 270 + len(anaList) // 10 * 280

		self.setMinimumSize(QtCore.QSize(width, 420))
		self.setMaximumHeight(420)

		anaList.reverse()

		for i, anaFolder in enumerate(anaList):
			name = anaFolder.split('/')[-2]
			size = 230

			self.btn_detect = QPushButton(name, self)
			self.btn_detect.move(20 + i // 10 * (size + 50), 20 + i % 10 * 40)
			self.btn_detect.resize(size, 30)
			self.btn_detect.setObjectName(anaFolder)
			self.btn_detect.clicked.connect(self.analyse)

	def analyse(self):
		sending_button = self.sender()
		anaFolder = str(sending_button.objectName())

		self.ana(anaFolder)
