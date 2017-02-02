# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:26:41 2016

@author: Gyutae Ha
"""

import cv2
import ibf


img1_fpath = './images/img1.jpg'
img2_fpath = './images/img2.png'

img1 = cv2.imread(img1_fpath, 0)
img2 = cv2.imread(img2_fpath, 0)

finder = ibf.IBF(img1, img2, ibf.MEDIUM_SIZE)
finder.getTransition()
finder.registerTwoImages(None)
