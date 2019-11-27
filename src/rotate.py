import os
import numpy as np
import cv2
import imutils

#path = 'img/_pre/'
path = 'img/sample/'
filenames = os.listdir(path)
 
label = []
out = []
for file in filenames:
	a = cv2.imread(path + file)
#	a = cv2.resize(a, (1000,1000))

	rows, cols, channels = a.shape
	for n in range(0,360,90) :
		rotate2 = imutils.rotate_bound (a, n)
#		rotation = cv2.getRotationMatrix2D((rows / 2, cols / 2), n, 0.5)
#		b = cv2.warpAffine(a, rotation, (rows, cols))    
		out.append(rotate2)
		cv2.imwrite('img/out/'+file+str(n)+'.jpg', rotate2)
#		out.append(b)
#		cv2.imwrite('img/out/'+file+str(n)+'.jpg', b)

