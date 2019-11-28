from PyQt5.QtWidgets import *

import sys
import os
import src.ui as ui
import src.preprocessing as pre
import src.trainer as tr
import src.crawler as cr
import yolo_video as yolo

if __name__ == '__main__':
	imgPath = "./img/"

	fileList = os.listdir(imgPath)
	csvFileList = []

	for file in fileList:
		if pre.getExt(file) == "csv":
			csvFileList.append(file)
	
	# preprocess()

	app = QApplication(sys.argv)
	mywindow = ui.MyWindow(csvFileList, yolo.detect_img)
	mywindow.show()
	app.exec_()

def preprocess():
	prePath = "./img/_pre/"
	etcPath = "./img/_etc/"

	for csvFile in csvFileList:
		brand = csvFile[:-4]
		dirPath = imgPath + brand + "/"
		pre.optimizeFolder(dirPath, etcPath, 1)
		pre.csvFloat2Int(dirPath)
		pre.saveCroppedImg(dirPath, prePath)
	pre.voTTCSV2YOLOAnnoTxt(imgPath, csvFileList)
	pre.plotTrainBoxRatio() # 전체 이미지에서 학습 영역이 차지하는 비율을 구해서 그래프로 그림