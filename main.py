import os
import src.ui as ui
import src.preprocessing as pre
import src.trainer as tr
import src.crawler as cr


def preprocess():
	imgPath = "./img/"
	prePath = "./img/_pre/"
	etcPath = "./img/_etc/"

	fileList = os.listdir(imgPath)
	csvFileList = []

	for file in fileList:
		if pre.getExt(file) == "csv":
			csvFileList.append(file)

	# for csvFile in csvFileList:
	# 	brand = csvFile[:-4]
	# 	dirPath = imgPath + brand + "/"
	# 	pre.optimizeFolder(dirPath, etcPath, 1)
	# 	pre.csvFloat2Int(dirPath)
	# 	pre.saveCroppedImg(dirPath, prePath)
	pre.voTTCSV2YOLOAnnoTxt(imgPath, csvFileList)
	# pre.plotTrainBoxRatio() # 전체 이미지에서 학습 영역이 차지하는 비율을 구해서 그래프로 그림

# preprocess()

cr.crawl("jonson131214@gmail.com", "q1w2e3r4!@", "나스", 50, 0)
# cr.crawl("qweewq1111@naver.com","zzzzzzzz11", "나스", 5, 1)
# tr.train()
# ui.run()
