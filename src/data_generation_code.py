#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1. 회전
# 2. 확대 or 축소

# 1. 변환해야 할 이미지가 있는 폴더
# 2. 변환된 이미지를 저장할 폴더

# openCV 축소하거나 회전시킬때 빈공간을 채워주는 것

import os
import numpy as np
import cv2

def train():
    path = 'img/img_pre/'
    filenames = os.listdir(path)
    
    label = []
    c = []
    
    for file in filenames:
        img = cv2.imread(path + file)
        img = cv2.resize(img, dsize=(80, 80))
        rows, cols, channels = img.shape
        
        # 45 degredd -> zoom 0.7
        mList = []
        mList.append(cv2.getRotationMatrix2D((rows/2, cols/2), -30, 0.75))
        mList.append(cv2.getRotationMatrix2D((rows/2, cols/2), -20, 0.8))
        mList.append(cv2.getRotationMatrix2D((rows/2, cols/2), -10, 0.85))
        mList.append(cv2.getRotationMatrix2D((rows/2, cols/2), 0, 1))
        mList.append(cv2.getRotationMatrix2D((rows/2, cols/2), 10, 0.85))
        mList.append(cv2.getRotationMatrix2D((rows/2, cols/2), 20, 0.8))
        mList.append(cv2.getRotationMatrix2D((rows/2, cols/2), 30, 0.75))
        
        dstList = []
        for no in range(len(mList)):
            dst = cv2.warpAffine(img, mList[no], (rows, cols))
            cv2.imwrite('img/img_save/' + file[:-4] + str(no) + file[-4:], dst)
            
        cv2.waitKey()
        cv2.destroyAllWindows()
        
train()

