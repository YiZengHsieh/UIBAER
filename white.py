# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:38:07 2020

@author: swz84
"""

import cv2
import numpy as np
import random
 
def white_balance_5(img):
    '''
    動態閾值算法
    算法分爲兩個步驟：白點檢測和白點調整。
    只是白點檢測不是與完美反射算法相同的認爲最亮的點爲白點，而是通過另外的規則確定
    :param img: cv2.imread讀取的圖片數據
    :return: 返回的白平衡結果圖片數據
    '''

    b, g, r = cv2.split(img)
    """
    YUV空間
    """
    def con_num(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        if x == 0:
            return 0
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    (y, u, v) = cv2.split(yuv_img)
    # y, u, v = cv2.split(img)
    m, n = y.shape
    sum_u, sum_v = 0, 0
    max_y = np.max(y.flatten())
    print(max_y)
    for i in range(m):
        for j in range(n):
            sum_u = sum_u + u[i][j]
            sum_v = sum_v + v[i][j]

    avl_u = sum_u / (m * n)
    avl_v = sum_v / (m * n)
    du, dv = 0, 0
    # print(avl_u, avl_v)
    for i in range(m):
        for j in range(n):
            du = du + np.abs(u[i][j] - avl_u)
            dv = dv + np.abs(v[i][j] - avl_v)

    avl_du = du / (m * n)
    avl_dv = dv / (m * n)
    num_y, yhistogram, ysum = np.zeros(y.shape), np.zeros(256), 0
    radio = 0.5  # 如果該值過大過小，色溫向兩極端發展
    for i in range(m):
        for j in range(n):
            value = 0
            if np.abs(u[i][j] - (avl_u + avl_du * con_num(avl_u))) < radio * avl_du or np.abs(
                    v[i][j] - (avl_v + avl_dv * con_num(avl_v))) < radio * avl_dv:
                value = 1
            else:
                value = 0

            if value <= 0:
                continue
            num_y[i][j] = y[i][j]
            yhistogram[int(num_y[i][j])] = 1 + yhistogram[int(num_y[i][j])]
            ysum += 1
    # print(yhistogram.shape)
    sum_yhistogram = 0
    # hists2, bins = np.histogram(yhistogram, 256, [0, 256])
    # print(hists2)
    Y = 255
    num, key = 0, 0
    while Y >= 0:
        num += yhistogram[Y]
        if num > 0.01 * ysum:    # 取前10%的亮點爲計算值，如果該值過大易過曝光，該值過小調整幅度小
            key = Y
            break
        Y = Y - 1
    # print(key)
    sum_r, sum_g, sum_b, num_rgb = 0, 0, 0, 0
    for i in range(m):
        for j in range(n):
            if num_y[i][j] > key:
                sum_r = sum_r + r[i][j]
                sum_g = sum_g + g[i][j]
                sum_b = sum_b + b[i][j]
                num_rgb += 1

    avl_r = sum_r / num_rgb
    avl_g = sum_g / num_rgb
    avl_b = sum_b / num_rgb
    print(avl_r,avl_g,avl_b)
    for i in range(m):
        for j in range(n):
            b_point = int(b[i][j]) * int(max_y) / avl_b
            g_point = int(g[i][j]) * int(max_y) / avl_g
            r_point = int(r[i][j]) * int(max_y) / avl_r
            #b_point = int(b[i][j]) / int(max_y)
            #g_point = int(g[i][j]) / int(max_y)
            #r_point = int(r[i][j]) / int(max_y)
            #print(b[i][j])
            if b_point>255:
                b[i][j] = 255
            else:
                if b_point<0:
                    b[i][j] = 0
                else:
                    b[i][j] = b_point
            if g_point>255:
                g[i][j] = 255
            else:
                if g_point<0:
                    g[i][j] = 0
                else:
                    g[i][j] = g_point
            if r_point>255:
                r[i][j] = 255
            else:
                if r_point<0:
                    r[i][j] = 0
                else:
                    r[i][j] = r_point

    return cv2.merge([b, g, r])

img = cv2.imread("D:/DCP/Jc0209115155.jpg")
whiteimg = white_balance_5(img)
cv2.imwrite('D:/DCP/white1sigsi511515151520022.jpg',whiteimg)
cv2.imwrite('D:/DCP/Output/result/result.jpg',whiteimg)