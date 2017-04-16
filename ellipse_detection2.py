# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:17:51 2017
1实现控制点提取
2对于具体问题我们具体分析，对于这个场景的椭圆检测，传统方法速度很慢，所以这个
2方法是一种根据实际应用而想到的，因为我们最终只是提出圆盘上的椭圆，而圆盘和整
2个背景反差较大，所以最终我选择了先把这个大的外轮廓抠出来，二值化后再进行小轮
2廓提取，最后进行椭圆拟合得到椭圆的参数
@author: sss
"""
import cv2
import numpy as np
from numba import jit
'''在contours中寻找最长的边缘'''
def findlongcontours(contours):
    a=0
    b=-1
    for i in range(len(contours)):
        if len(contours[i])>a:
            a=len(contours[i])
            b=i
    return b

@jit
def selectarea(img,img_copy):
    emptyImage=np.zeros(img.shape,np.uint8)
    emptyImage[...]=255
    rows,cols,channels=img.shape
    for i in range(rows):
        for j in range(cols):
            if img_copy[i,j][0]==0 and img_copy[i,j][1]==0 and img_copy[i,j][2]==255:
                emptyImage[i,j]=img[i,j]
    return emptyImage
            
def getellipses(filename):
    img=cv2.imread(filename)
    img_copy=cv2.imread(filename)
    img_g=cv2.imread(filename,0)
    ret,img_b=cv2.threshold(img_g,160,255,0)
    img_f,contours, hierarchy = cv2.findContours(img_b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_copy,contours,findlongcontours(contours),(0,0,255),-1)
    img_select=selectarea(img,img_copy)
    img_sgray=cv2.cvtColor(img_select,cv2.COLOR_BGR2GRAY)
    ret,img_point=cv2.threshold(img_sgray,80,255,1)
    img_f,contours, hierarchy = cv2.findContours(img_point,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    ellipses=[]
    for each in contours:
        if len(each)>50:
            ellipse=cv2.fitEllipse(each)
            ellipses.append(ellipse)
    return ellipses


    
    
    
path='E:\\computer vision\\Image\\Buddha_00'
ellipses_all=[]
for i in range(2):
    filename=path+str(i+1)+'.jpg'
    ellipses_all.append(getellipses(filename))
    

    