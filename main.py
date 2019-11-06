import os
import src.preprocessing as pre


imgPath = "./img/"
dirPath = "./img/chanel/"
outPath = "./img/_pre/"
etcPath = "./img/_etc/"

pre.saveCroppedImg(dirPath, outPath)

fileList = os.listdir(imgPath)
folderList = []

for file in fileList:
        if pre.getExt(file) == "csv":
            folderList.append(file[:-4])

for folderName in folderList:
	if folderName == "drjart":
		pre.optimizeFolder(dirPath, etcPath, 1)
		break
	pre.saveCroppedImg(folder, path, outPath)
	# pre.voTTCSV2YOLOAnnoTxt()
	pre.plotTrainBoxRatio() # 전체 이미지에서 학습 영역이 차지하는 비율을 구해서 그래프로 그림