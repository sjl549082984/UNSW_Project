import cv2
import numpy as np

# this file is Q5

img1 = cv2.imread('imageQ5.png', 0)

minmax_img = np.zeros((img1.shape[0], img1.shape[1]), dtype='uint8')

for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        minmax_img[i, j] = 255 * (img1[i, j] - np.min(img1)) / (np.max(img1) - np.min(img1))
median = cv2.medianBlur(img1, 3)
equ = cv2.equalizeHist(median)

kernel = np.array([[0.0, -1.0, 0.0],
                   [-1.0, 5.0, -1.0],
                   [0.0, -1.0, 0.0]])

kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

img_rst = cv2.filter2D(equ,-1,kernel)
res = np.hstack((img1,img_rst))
#cv2.imshow('origin', img1)
#cv2.imshow('1', minmax_img)
#cv2.imshow('2', median)
#cv2.imshow('3', equ)
#cv2.imshow('4', img_rst)
cv2.imshow('final', res)

cv2.waitKey(-1)