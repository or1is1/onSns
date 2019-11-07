import os
import preprocessing as pre
import crawler as cr


def preprocess():
	imgPath = "./img/"
	prePath = "./img/_pre/"
	etcPath = "./img/_etc/"

	fileList = os.listdir(imgPath)
	csvFileList = []

	for file in fileList:
		if pre.getExt(file) == "csv":
			csvFileList.append(file)

	for csvFile in csvFileList:
		dirPath = imgPath + csvFile[:-4] + "/"
		# pre.optimizeFolder(dirPath, etcPath, 1)
		pre.csvFloat2Int(dirPath)
		pre.saveCroppedImg(dirPath, prePath)

	pre.voTTCSV2YOLOAnnoTxt(imgPath, csvFileList)
	# pre.plotTrainBoxRatio() # 전체 이미지에서 학습 영역이 차지하는 비율을 구해서 그래프로 그림

# preprocess()
cr.crawl("립스틱", "10", 0)