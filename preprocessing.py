import numpy as np
import pandas as pd
import cv2
import os


imgFormatList = ["jpg", "jpeg", "png"]

def getFilenameWoExt(Filename):
    return ".".join(Filename.split('.')[:-1])

def getExt(Filename):
    return Filename.split('.')[-1].lower()

def getCSVName(dirPath):
    if dirPath[-1] == '/':
        return dirPath[:-1] + ".csv"
    else:
        return dirPath + ".csv"

def getBrand(dirPath):
    return dirPath[:-1].split('/')[-1]

def optimizeFolder(dirPath, etcPath, flag):
    """
    flag = 0 means labelImg Program
    flag = 1 means VoTT Program
    """
    if not os.path.isdir(etcPath):
        os.mkdir(etcPath)

    count = 0
    imgFilenameList = []
    txtFilenameList = []
    etcFilenameList = []

    if flag == 0: # labelImg
        # 텍스트 파일과 매칭되지 않는 파일 삭제
        FilenamesList = os.listdir(dirPath)
        
        for Filename in FilenamesList:
            ext = getExt(Filename)

            if ext == "txt":
                txtFilenameList.append(getFilenameWoExt(Filename))
            else:
                etcFilenameList.append(Filename)

        for etcFilename in etcFilenameList:
            if getFilenameWoExt(etcFilename) not in txtFilenameList:
                os.replace(dirPath + etcFilename, etcPath + etcFilename)
                count += 1

        txtFilenameList.clear()
        etcFilenameList.clear()

        # 이미지 파일과 매칭되지 않는 파일 삭제
        FilenamesList = os.listdir(dirPath)

        for Filename in FilenamesList:
            ext = getExt(Filename)

            if ext in imgFormatList:
                imgFilenameList.append(getFilenameWoExt(Filename))
            else:
                etcFilenameList.append(Filename)

        for etcFilename in etcFilenameList:
            if getFilenameWoExt(etcFilename) not in imgFilenameList:
                os.replace(dirPath + etcFilename, etcPath + etcFilename)
                count += 1

        imgFilenameList.clear()
        etcFilenameList.clear()
    elif flag == 1: # VoTT
        # csv 파일과 매칭되지 않는 이미지 삭제
        csvFileName = getCSVName(dirPath)
        df = pd.read_csv(csvFileName)
        
        FilenamesList = os.listdir(dirPath)

        for Filename in FilenamesList:
            if Filename not in df[['image']].values:
                os.replace(dirPath + Filename, etcPath + Filename)
                count += 1

        # 이미지 파일과 매칭되지 않는 csv row 삭제
        FilenamesList = os.listdir(dirPath)
        
        etcDf = pd.DataFrame(columns=df.columns)

        for imgName in df[['image']].values:
            imgName = imgName[0]
            
            if imgName not in FilenamesList:
                etcDf = etcDf.append(df[df.image == imgName])
                df = df[df.image != imgName]
                count += 1

        df.to_csv(csvFileName, index=False)
        etcDf.to_csv(etcPath + getCSVName(getBrand(dirPath)), index=False)
    else:
        raise Exception("Not supported flag Exception")

    print("deleted {} file(s)".format(count))

def labelImgYOLO2VoTTCSV(dirPath, etcPath, label):
    FilenameList = os.listdir(dirPath)
    csvFileName = getCSVName(dirPath)
    imgFilenameList = []
    imgname = None

    with open(csvFileName, 'w', errors='ignore') as out:
        out.write('"image","xmin","ymin","xmax","ymax","label"\n')

        for Filename in FilenameList:
            if getExt(Filename) in imgFormatList:
                imgFilenameList.append(Filename)

        for imgFilename in imgFilenameList:
            txtFilename = getFilenameWoExt(imgFilename) + ".txt"
            img = cv2.imread(dirPath + imgFilename)
            try:
                img_y, img_x = img.shape[:2]
            except AttributeError as e:
                os.replace(dirPath + imgFilename, etcPath + imgFilename)
                os.replace(dirPath + txtFilename, etcPath + txtFilename)
                continue

            # with open(dirPath + txtFilename, 'r', errors='ignore') as textfile:
            with open(dirPath + txtFilename, 'r') as textfile:
                lines = textfile.readlines()
                for line in lines:
                    x, y, w, h = line.split()[1:]
                    x = float(x)
                    y = float(y)
                    w = float(w)
                    h = float(h)
                    out.write('"{}",{},{},{},{},{}\n'.format(imgFilename, x * img_x - (w * img_x / 2), y * img_y - (h * img_y / 2), x * img_x + (w * img_x / 2), y * img_y + (h * img_y / 2), label))

def csvFloat2Int(dirPath):
    csvFileName = getCSVName(dirPath)
    df = pd.read_csv(csvFileName)
    df[['xmin', 'ymin', 'xmax', 'ymax']] = df[['xmin', 'ymin', 'xmax', 'ymax']].astype('uint16')
    df.to_csv(csvFileName, index = False)

def saveCroppedImg(dirPath, outPath):
    """
    'imgPath' is image folder path
    """
    brand = getBrand(dirPath)
    csvFileName = getCSVName(dirPath)
    df = pd.read_csv(csvFileName)
        
    if not os.path.isdir(outPath):
        os.mkdir(outPath)
    
    for i in range(len(df)):
        sr = df.iloc[i]
        filePath = dirPath + sr['image']
        img = cv2.imread(filePath)

        if img is not None:
            croppedimg = img[sr['ymin']:sr['ymax'], sr['xmin']:sr['xmax']]
            cv2.imwrite("{}{}_{}_{}.png".format(outPath, sr['label'], brand, i,), croppedimg)
            # cv2.rectangle(img, (sr['xmin'], sr['ymin']), (sr['xmax'], sr['ymax']), (0, 0, 255))
            # cv2.imshow("img", img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
        else:
            print("****************************************")
            print("*", filePath, "is None")
            print(sr)
            print("****************************************")

        if i % 100 == 0 and i != 0:
            print("{} / {} = {}%".format(i, len(df), int(i / len(df) * 10000) / 100))

    print()

def voTTCSV2YOLOAnnoTxt(imgPath, csvFileList):
    outList = []

    for csvFile in csvFileList:
        prvImgName = None

        with open(imgPath + csvFile) as csvFileName:
            lines = csvFileName.readlines()
            lines = lines[1:] # 컬럼 제거

            for line in lines:
                splitted = line.split(',')
                splitted[-1] = splitted[-1].replace('"', '') # remove double quote
                splitted[-1] = splitted[-1].replace('\n', '') # delete newline
                splitted[1:] = list(map(float, splitted[1:])) # string to float
                splitted[1:] = list(map(int, splitted[1:])) # float to int
                splitted[1:] = list(map(str, splitted[1:])) # int to str

                if prvImgName != splitted[0]:
                    splitted[1] = imgPath + csvFile[:-4] + "/" + splitted[0].replace('"', '') # remove double quote
                    splitted[1] = " ".join(splitted[1:3]) # concat
                    splitted[1] = ",".join(splitted[1:]) # concat
                    outList.append(splitted[1])
                else:
                    outList[-1] = outList[-1] + " " + ",".join(splitted[1:]) # concat

                prvImgName = splitted[0]

    with open("./train.txt", 'w') as outfile:
        for line in outList:
            outfile.write(line + '\n')
