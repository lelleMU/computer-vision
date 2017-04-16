# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 22:12:53 2017

@author: lelle
"""
from numba import jit
import cv2
import numpy as np
import math
def dis(x1,y1,x2,y2):
    dis=math.sqrt((x1-x2)**2+(y1-y2)**2)
    return dis
class ellipse:
    def __init__(self,a,b,pos,alpha):
        self.a=a
        self.b=b
        self.pos=pos
        self.alpha=alpha
#读取图像
img=cv2.imread('E:\\Image\\text2.jpg',0)
#对图像做边缘提取
img_canny=cv2.Canny(img,200,300)
#获取所有边缘点行列号
@jit
def ellipse_detection(img,mina,maxa,min_v): 
    #ellipses=[]
    px,py=np.where(img)
    acc=[0]*max(img.shape)
    for i in range(len(px)):
        for j in range(len(px)-1,i,-1):
            x1=px[i]
            y1=py[i]
            x2=px[j]
            y2=py[j]
            acc=[0]*max(img.shape)#累加器置零
            d12=dis(x1,y1,x2,y2)
            a=d12/2
            if a>mina and a<maxa:
                x0=(x1+x2)/2
                y0=(y1+y2)/2
                alpha=math.atan2(x2-x1,y2-y1)
                #遍历其他点
                for m in range(len(px)):
                    #第三点不能是上面的两个点
                    if m==i or m==j:
                        continue
                    x3=px[m]
                    y3=py[m]
                    d03=dis(x0,y0,x3,y3)
                    
                    if d03>=a or d03==0:
                        continue
                    d23=dis(x2,y2,x3,y3)
                    cos_a=(d03**2+a**2-d23**2)/(2*d03*a)
                    xx=(cos_a**2)*(d03**2)
                    yy=(1-cos_a**2)*(d03**2)
                    if  yy<=0:
                        continue
                    b=round(math.sqrt(((a**2)*yy)/(a**2-xx)))
                    acc[b]+=1
                votes=max(acc)
                if votes>min_v: #投票数超过阈值，则找到一个椭圆
                    b=acc.index(votes)
                    ep=ellipse(a,b,[x0,y0],alpha)
                    #ellipses.append(ep)
                    epr=int(round(ep.pos[0]))
                    epc=int(round(ep.pos[1]))
                    if(img_canny[epr,epc]<255):   
                        img_canny[epr,epc]+=1
                    
ellipse_detection(img_canny,5,100,10)
cv2.namedWindow("001",cv2.WINDOW_NORMAL)
cv2.imshow('001',img_canny)
cv2.waitKey(0)