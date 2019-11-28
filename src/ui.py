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

	def __init__(self, csvFileList, detect_img):
		super().__init__()
		self.crawlDict = {'인스타그램':0, '핀터레스트':0}
		self.detect_img = detect_img

		self.instavalue = 0
		self.pintervalue = 0

		self.setWindowTitle("키워드에 따른 브랜드 통계")
		self.setMinimumSize(QtCore.QSize(410, 210))
		self.setMaximumHeight(200)

		self.centralwidget = QtWidgets.QWidget(self)
		self.centralwidget.setObjectName("centralwidget")

		self.lineEdit = QLineEdit("keyword", self)
		self.lineEdit.move(20, 20)
		self.lineEdit.resize(170, 30)
		self.lineEdit.returnPressed.connect(self.actionstart)

		self.count = QLineEdit("100", self)
		self.count.move(200, 20)
		self.count.resize(30, 30)
		self.count.returnPressed.connect(self.actionstart)

		self.btn1 = QPushButton("크롤링", self)
		self.btn1.move(240, 20)
		self.btn1.resize(80, 30)
		self.btn1.clicked.connect(self.actionstart)

		self.btn2 = QPushButton("분석", self)
		self.btn2.move(330, 20)
		self.btn2.resize(60, 30)
		self.btn2.clicked.connect(self.analyse)

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
			save_dir = os.getcwd() + "\\out"
			if not os.path.isdir(save_dir):
				os.makedirs(os.path.join(save_dir))
			subprocess.call("explorer {}".format(save_dir), shell=True)
			for i, sns in enumerate(self.crawlDict.values()):
				if sns == 1 :
					count = int(self.count.text())
					if i == 0:
						cr.crawl("jonson131214@gmail.com", "q1w2e3r4!@", keyword, count, 0)
					elif i == 1:
						cr.crawl("qweewq1111@naver.com", "zzzzzzzz11", keyword, count, 1)

	def analyse(self) :
		path = "./out/"
		folderList = os.listdir(path)
		anaList = []

		for folder in folderList:
			if os.path.isfile(path + folder + "/url.csv"):
				anaList.append(path + folder + "/")
				
		for anaFolder in anaList:
			print(anaFolder)
		# self.detect_img(YOLO())
