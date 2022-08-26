# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:43:26 2020

@author: swz84
"""

import cv2
import numpy as np

def LSAC(Ib, Ig, Ir, blockSize,ab,ag,ar,p):
    
    addSize = int(blockSize / 2)
    newHeight = img.shape[0] + blockSize
    newWidth = img.shape[1] + blockSize
    
    oldab = np.zeros((newHeight, newWidth))
    oldag = np.zeros((newHeight, newWidth))
    oldar = np.zeros((newHeight, newWidth))
    oldab = ab
    oldag = ag
    oldar = ar
    imgbMiddle = np.zeros((newHeight, newWidth))
    imggMiddle = np.zeros((newHeight, newWidth))
    imgrMiddle = np.zeros((newHeight, newWidth))
    
    imgbDark = np.zeros((img.shape[0], img.shape[1]))
    imggDark = np.zeros((img.shape[0], img.shape[1]))
    imgrDark = np.zeros((img.shape[0], img.shape[1]))
    
    imgab = np.zeros((img.shape[0], img.shape[1]))
    imgag = np.zeros((img.shape[0], img.shape[1]))
    imgar = np.zeros((img.shape[0], img.shape[1]))
    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            a = 5
            if i-addSize == 0 or i+addSize == newHeight-1:
                a = a-1
            if j-addSize == 0 or j+addSize == newWidth-1:
                a = a-1      
            imgbDark[i - addSize, j - addSize] = (ab[i,j]+ab[i-1,j]+ab[i+1,j]+ab[i,j-1]+ab[i,j+1])/a
            imggDark[i - addSize, j - addSize] = (ag[i,j]+ag[i-1,j]+ag[i+1,j]+ag[i,j-1]+ag[i,j+1])/a
            imgrDark[i - addSize, j - addSize] = (ar[i,j]+ar[i-1,j]+ar[i+1,j]+ar[i,j-1]+ar[i,j+1])/a
    imgab[:,:] = (Ib[:,:]*p)+(imgbDark[:,:]*(1-p))
    imgag[:,:] = (Ig[:,:]*p)+(imggDark[:,:]*(1-p))
    imgar[:,:] = (Ir[:,:]*p)+(imgrDark[:,:]*(1-p))
    
    imgbMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = imgab
    imggMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = imgag
    imgrMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = imgar
    lossb = np.abs(np.sum(imgab)-np.sum(oldab))/(imgab.shape[0]*imgab.shape[1])
    lossg = np.abs(np.sum(imgag)-np.sum(oldag))/(imgab.shape[0]*imgab.shape[1])
    lossr = np.abs(np.sum(imgar)-np.sum(oldar))/(imgab.shape[0]*imgab.shape[1])
    
    return imgbMiddle,imggMiddle,imgrMiddle,imgab,imgag,imgar,lossb,lossg,lossr

img = cv2.imread('D:/DCP/InputImages/LFT_3374.jpg')
initab = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
initag = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
initar = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
(Ib,Ig,Ir) = cv2.split(img)
s = 488
sig = 0.08
p = 1/((sig**2)*(s**2)+1)
total_lossb = np.array([])
total_lossg = np.array([])
total_lossr = np.array([])
for i in range(1000):
    if i == 0:
        imgbMiddle,imggMiddle,imgrMiddle,imgab,imgag,imgar,lossb,lossg,lossr = LSAC(Ib,Ig,Ir,2,initab,initag,initar,0.001)
        #total_lossb = np.array([lossb])
    else:
        imgbMiddle,imggMiddle,imgrMiddle,imgab,imgag,imgar,lossb,lossg,lossr = LSAC(Ib,Ig,Ir,2,imgbMiddle,imggMiddle,imgrMiddle,0.001)
    total_lossb = np.append(total_lossb,lossb)
    total_lossg = np.append(total_lossg,lossg)
    total_lossr =np.append(total_lossr,lossr)
    #print(total_lossb,total_lossg,total_lossr)
    
imgdark = cv2.merge([imgab,imgag,imgar])
imgdark = imgdark*2
cv2.imwrite('D:/DCP/LSAregg3374.jpg',imgdark)