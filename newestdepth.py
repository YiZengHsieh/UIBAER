# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:05:12 2020

@author: swz84
"""
from GuidedFilter import GuidedFilter
import skimage.io as io
from skimage import data_dir,io,transform,color
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.linear_model import LinearRegression


def getMinChannel(img):
    
    imgGray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    localMin = 255
    minchannel = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            localMin = 255
            for k in range(0, 3):
                if img.item((i, j, k)) < localMin:
                    localMin = img.item((i, j, k))
                    minchannel[i,j] = k
            imgGray[i, j] = localMin
    #print(minchannel)
    return imgGray,minchannel


def getDarkChannel(Ib, Ig, Ir, blockSize, Sb, Sg, Sr, Wb, Wg, Wr):
    
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1
    
    #print(Ir)

    imgbMiddle = np.zeros((newHeight, newWidth))
    imggMiddle = np.zeros((newHeight, newWidth))
    imgrMiddle = np.zeros((newHeight, newWidth))
    #mincmiddle = np.zeros((newHeight, newWidth))
    imgbMiddle[:, :] = 1
    imggMiddle[:, :] = 1
    imgrMiddle[:, :] = 1
    # print('imgMiddle',imgMiddle)
    # print('type(newHeight)',type(newHeight))
    # print('type(addSize)',type(addSize))
    imgbMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = Ib
    imggMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = Ig
    imgrMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = Ir
    #mincmiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = minchannel
    # print('imgMiddle', imgMiddle)
    imgbDark = np.zeros((img.shape[0], img.shape[1]))
    imggDark = np.zeros((img.shape[0], img.shape[1]))
    imgrDark = np.zeros((img.shape[0], img.shape[1]))
    #newminchannel = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    localMinb = 255
    localMing = 255
    localMinr = 255
    for i in range(addSize, newHeight - addSize):
        
        for j in range(addSize, newWidth - addSize):
            localMinb = 1
            localMing = 1
            localMinr = 1
            for k in range(i - addSize, i + addSize + 1):
                for l in range(j - addSize, j + addSize + 1):
                    itemb = imgbMiddle.item((k,l))
                    itemg = imggMiddle.item((k,l))
                    itemr = imgrMiddle.item((k,l))
                    #print(1 - Wb*abs(Sb-itemb))
                    if 1 - Wb*abs(Sb-itemb) < localMinb:
                        localMinb = 1 - Wb*abs(Sb-itemb)
                    elif 1 - Wg*abs(Sg-itemg) < localMing:
                        localMing = 1 - Wg*abs(Sg-itemg)
                    elif 1 - Wr*abs(Sr-itemr) < localMing:
                        localMinr = 1 - Wr*abs(Sr-itemr)
                        #localminc = mincmiddle.item((k,l))
                        #print(localminc)
            #newminchannel[i - addSize, j - addSize] = localminc
            #print(localMinb)
            imgbDark[i - addSize, j - addSize] = localMinb
            imggDark[i - addSize, j - addSize] = localMing
            imgrDark[i - addSize, j - addSize] = localMinr
    #print(img)
    print(imgrDark)
    return imgbDark,imggDark,imgrDark
file = open(r'D:/DCP/filepath.txt')
path = file.readline()
#print(path)
img = cv2.imread(path)
#grayimg = cv2.imread('D:/DCP/test4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#grayimg = cv2.cvtColor(grayimg, cv2.COLOR_BGR2GRAY)
kernelx = np.array([[1, 0, -1], 
                    [2, 0, -2], 
                    [1, 0, -1]])
kernely = np.array([[1, 2, 1], 
                    [0, 0, 0], 
                    [-1, -2, -1]])
SOBELX=cv2.filter2D(gray,-1,kernelx)
SOBELY=cv2.filter2D(gray,-1,kernely)
#G=np.sqrt(np.square(SOBELX)+np.square(SOBELY))
G=abs(SOBELX)+abs(SOBELY)
#G=G.astype(np.uint8)
print(G)
print(np.max(G),np.min(G))
window = np.ones((7,7), np.uint8)
window2 = np.ones((8,8), np.uint8)
img_dilation = cv2.dilate(G, window, iterations=1)
img_erosion = cv2.erode(img_dilation, window2, iterations=1) 
img_median=cv2.medianBlur(img_erosion,5)
a=np.max(img_median)-np.min(img_median)
print(a)
g = np.zeros(img_median.shape)
print(img_median.shape)
for i in range (img_median.shape[0]):
    for j in range (img_median.shape[1]):
        g[i][j]=(img_median[i][j]-np.min(img_median))/a
print(np.max(g),np.min(g))
    #Dr
Dr=np.zeros(g.shape)
Dr=(1-g)*255
Dr=np.around(Dr)
cv2.imwrite('D:/DCP/roughdepthabs12354453.jpg',Dr)
print('Dr',Dr)

regDr=np.array(Dr).reshape(Dr.shape[0]*Dr.shape[1],1)
#print('regDr',regDr)    
(Ib,Ig,Ir) = cv2.split(img)
regIb= np.array(Ib).reshape(Ib.shape[0]*Ib.shape[1],1)
regIg= np.array(Ig).reshape(Ig.shape[0]*Ig.shape[1],1)
regIr= np.array(Ir).reshape(Ir.shape[0]*Ir.shape[1],1)
regDr = regDr/255
regIb = regIb/255
regIg = regIg/255
regIr = regIr/255
print('regIb',regIb)
modelr=LinearRegression()
Rreg=modelr.fit(regDr,regIr)
Ar=modelr.coef_
#xrfit = np.linspace(0, 1, 437400)
'''
modelr.predict(regDr)
plt.scatter(regDr, regIr,color='red',s = 1,alpha=0.6)
plt.plot(regDr, modelr.predict(regDr),color='orange',linewidth =3)
plt.show()
'''
modelg=LinearRegression()
Greg=modelg.fit(regDr,regIg)
Ag=modelg.coef_
'''
modelg.predict(regDr)
plt.scatter(regDr, regIg,color='green',s = 0.1,alpha=0.6)
plt.plot(regDr, modelg.predict(regDr),color='orange',linewidth =3)
plt.show()    
'''
modelb=LinearRegression()
Breg=modelb.fit(regDr,regIb)
Ab=modelb.coef_
'''
modelb.predict(regDr)
plt.scatter(regDr, regIb,color='blue',s = 0.1,alpha=0.6)
plt.plot(regDr, modelb.predict(regDr),color='orange',linewidth =3)
plt.show()    
'''
print('Ac:',Ar,Ag,Ab)

if Ar > 0:
    Sr = 1
else:
    Sr = 0
        
if Ag > 0:
    Sg = 1
else:
    Sg = 0
    
if Ab > 0:
    Sb = 1
else:
    Sb = 0
print('Sc:',Sr,Sg,Sb)
#imgGray,minchannel = getMinChannel(img)
#imgdark = imgdark/255
Wr = math.tanh(4*abs(Ar))
Wg = math.tanh(4*abs(Ag))
Wb = math.tanh(4*abs(Ab))
imgbdark,imggdark,imgrdark = getDarkChannel(Ib/255, Ig/255, Ir/255, 5, Sb, Sg, Sr, Wb, Wg, Wr)
imgdark = cv2.merge([imgbdark,imggdark,imgrdark])
newimgdark = imgdark*255
Dd = np.zeros((img.shape[0],img.shape[1]))
Dd[:,:] = 255
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(0,3):
            if newimgdark[i,j,k] < Dd[i,j]:
                Dd[i,j] = newimgdark[i,j,k]

Dd = Dd.astype(np.uint8)
#print(Dd)
#Dd255 = np.zeros((img.shape[0],img.shape[1]))
#Dd255[:,:] = 255
#newDd = abs(Dd-Dd255)
#newDd = newDd.astype(np.uint8)
#print(newDd)
Dd_median=cv2.medianBlur(Dd,5)
#guided_filter = GuidedFilter(img, 50, 10 ** -3)
#NminDD_median = guided_filter.filter(Dd)
#Dd_median = np.abs(Dd_median-Dd255)
cv2.imwrite('D:/DCP/testdepthabs_020912132.jpg',Dd)
cv2.imwrite('D:/DCP/test24_0209123123.jpg',Dd_median)
cv2.imwrite('D:/DCP/Output/Depth/depth_map.jpg',Dd_median)
#cv2.imwrite('filter.jpg',NminDD_median)
#DDr = 1 - Wr*abs(Sr-Ir)
#DDg = 1 - Wg*abs(Sg-Ig)
#DDb = 1 - Wb*abs(Sb-Ib)
'''
minDD=np.zeros(Dr.shape)
for i in range (Dr.shape[0]):
    for j in range (Dr.shape[1]):
        if newminchannel[i,j] == 0:
            minDD[i,j] = 1- Wb * abs(Sb-imgdark[i,j])
        elif newminchannel[i,j] == 1:
            minDD[i,j] = 1- Wg * abs(Sg-imgdark[i,j])
        else:
            minDD[i,j] = 1- Wr * abs(Sr-imgdark[i,j])
'''
#newminDD = minDD*255
#NminDD_median = cv2.medianBlur(newminDD.astype(np.float32),5)
#guided_filter = GuidedFilter(img, 50, 10 ** -3)
#NminDD_median = guided_filter.filter(newminDD)
#cv2.imwrite('depthabs1.jpg',NminDD_median)
'''
nminDD=np.max(minDD)-np.min(minDD)
NminDD= np.zeros(minDD.shape)
for i in range (minDD.shape[0]):
    for k in range (minDD.shape[1]):
        NminDD[i][k]=((minDD[i][k]-np.min(minDD))/nminDD)*255
NminDD_median = cv2.medianBlur(NminDD.astype(np.float32),5)
cv2.imwrite('depthabs1.jpg',NminDD_median)
'''
