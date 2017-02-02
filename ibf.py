# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:51:33 2016

@author: Gyutae Ha
"""

import numpy as np
import cv2
import math
import numbers
import random

"""
IBF is abbreviation of 'Image Bug Finder'.
...

"""

SMALL_SIZE = 400 * 400
MEDIUM_SIZE = 600 * 600
LARGE_SIZE = 1000 * 1000

class IbfException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value

class Vec2d:
    
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], tuple):
            arg_tuple = args[0]
            if isinstance(arg_tuple[0], numbers.Number) and isinstance(arg_tuple[1], numbers.Number):
                self.x = arg_tuple[0]
                self.y = arg_tuple[1]
            else:
                self.x = 0
                self.y = 0
        elif len(args) == 2 and isinstance(args[0], numbers.Number) and isinstance(args[1], numbers.Number):
            self.x = args[0]
            self.y = args[1]
        else:
            self.x = 0
            self.y = 0

    def copyObj(self, vec2d):
        self.x = vec2d.x
        self.y = vec2d.y            
    
    def scalarProduct(self, c):
        self.x = self.x * c
        self.y = self.y * c
        
    def move(self, dx, dy):
        self.x = self.x + dx
        self.y = self.y + dy
        
    def rotate(self, theta):
        tmp_x = self.x
        tmp_y = self.y
        self.x = tmp_x * math.cos(theta) - tmp_y * math.sin(theta)
        self.y = tmp_x * math.sin(theta) + tmp_y * math.cos(theta)
        
    def getStr(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'
        
class IBF(object):
    
    """
    d_size   : target size; SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE
    trans    : translation & rotation values to match two images
    """    
    
    def __init__(self, template_img, target_img, desired_size):
        
        if isinstance(template_img, np.ndarray) and isinstance(desired_size, int) and desired_size > 10000:
            self.template_img = self._resizeImage(template_img)
            self.target_img = self._resizeImage(target_img)
            self.d_size = desired_size
            self.trans = {}     
        else:
            raise IbfException('Raised exception in IBF constructor.')
               
    def registerTwoImages(self, trans):
        """
        trans['rot_cent']
        trans['size_ratio']
        trans['rot_angle']
        trans['cent_trans']
        """
    
        if (trans == None):
            trans = self.trans
            
        res_img1 = cv2.resize(self.template_img, None, fx=trans['size_ratio'], fy=trans['size_ratio'])
        rows = res_img1.shape[0]
        cols = res_img1.shape[1]
        
        c_x = int(round(trans['rot_cent'].x * trans['size_ratio'], 0))
        c_y = int(round(trans['rot_cent'].y * trans['size_ratio'], 0))
       
        diag_vecs = []
        diag_vecs.append((cols - c_x, c_y))             # first quadrant
        diag_vecs.append((-c_x, c_y))                   # second quadrant
        diag_vecs.append((-c_x, c_y - rows))            # third quadrant
        diag_vecs.append((cols - c_x, c_y - rows))      # fourth quadrant
        
        rot_diag_vecs = []   
        for diag_vec in diag_vecs:
            rot_diag_vec = Vec2d(diag_vec)
            rot_diag_vec.rotate(trans['rot_angle'])
            rot_diag_vecs.append(rot_diag_vec)
    
        im_diag = math.sqrt(cols**2 + rows**2) 
        max_rot_diag_x = -im_diag
        min_rot_diag_x = im_diag
        max_rot_diag_y = -im_diag
        min_rot_diag_y = im_diag
        
        for rot_diag_vec in rot_diag_vecs:
            if rot_diag_vec.x > max_rot_diag_x:
                max_rot_diag_x = rot_diag_vec.x
            if rot_diag_vec.x < min_rot_diag_x:
                min_rot_diag_x = rot_diag_vec.x
            if rot_diag_vec.y > max_rot_diag_y:
                max_rot_diag_y = rot_diag_vec.y   
            if rot_diag_vec.y < min_rot_diag_y:
                min_rot_diag_y = rot_diag_vec.y
    
        right_diff_x = int(round(max_rot_diag_x - (cols - c_x), 0))
        left_diff_x = int(round(-c_x - min_rot_diag_x, 0))
        up_diff_y = int(round(max_rot_diag_y - c_y, 0))
        down_diff_y = int(round((c_y - rows) - min_rot_diag_y, 0))
        
        right_added_x = right_diff_x if right_diff_x > 0 else 0
        left_added_x = left_diff_x if left_diff_x > 0 else 0
        up_added_y = up_diff_y if up_diff_y > 0 else 0
        down_added_y = down_diff_y if down_diff_y > 0 else 0
    
        e_rows = rows + up_added_y + down_added_y
        e_cols = cols + left_added_x + right_added_x
        img2_rows = self.target_img.shape[0]
        img2_cols = self.target_img.shape[1]
        
        rst_rows = e_rows if e_rows > img2_rows + up_added_y + down_added_y else img2_rows + up_added_y + down_added_y
        rst_cols = e_cols if e_cols > img2_cols + left_added_x + right_added_x else img2_cols + left_added_x + right_added_x
        
        rst_img1 = np.zeros((rst_rows, rst_cols), np.uint8)
        rst_img2 = np.zeros((rst_rows, rst_cols), np.uint8)
        
        rst_map = np.zeros((rst_rows, rst_cols), np.uint8)
    
        rst_img1[up_added_y : up_added_y + rows, left_added_x : left_added_x + cols] = res_img1
        c_x += left_added_x
        c_y += up_added_y
        M = cv2.getRotationMatrix2D((c_x, c_y), trans['rot_angle'], 1)
        rst_img1 = cv2.warpAffine(rst_img1, M, (rst_cols, rst_rows))
        M = np.float32([[1, 0, trans['cent_trans'].x],[0, 1, trans['cent_trans'].y]])
        rst_img1 = cv2.warpAffine(rst_img1,M,(rst_cols,rst_rows))
    
    
        rst_img2[up_added_y : up_added_y + img2_rows, left_added_x : left_added_x + img2_cols] = self.target_img
    
        for i in range(0, rst_rows):
            for j in range(0, rst_cols):
                if getAbs(rst_img1[i][j], rst_img2[i][j]) > 20.0:
                    #pass
                    rst_map[i][j] = 255
                    
        print rst_img2[rst_rows/2][rst_cols/2]
    
        cv2.imshow("1", rst_img1)
        cv2.imshow("2", rst_img2)
        cv2.imshow("rst", rst_map)  
    
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                   
    def getTransition(self):
        trans_list = self._getTransitionListComparingMatchedSegments(self.template_img, self.target_img, 10)
        if trans_list == 0:
            return 0
        
        # select most freq value
        scale_freq = {}
        angle_freq = {}
        for trans in trans_list:
            #print trans
            size_ratio = trans['size_ratio']
            rot_angle = trans['rot_angle']
            if size_ratio in scale_freq:
                scale_freq[size_ratio] += 1
            else:
                scale_freq[size_ratio] = 1
            if rot_angle in angle_freq:
                angle_freq[rot_angle] += 1
            else:
                angle_freq[rot_angle] = 1
    
        most_freq = 0
        for key, val in scale_freq.iteritems():
            if val > most_freq:
                most_freq = val
                mf_scale = key
        
        most_freq = 0
        for key, val in angle_freq.iteritems():
            if val > most_freq:
                most_freq = val
                mf_angle = key
                       
        selected_trans = {}
        for trans in trans_list:
           if trans['size_ratio'] == mf_scale and trans['rot_angle'] == mf_angle:
               selected_trans = trans
               break

        self.trans = selected_trans        
        
        return selected_trans        
        
    def _resizeImage(self, img, tSize = MEDIUM_SIZE):
        w = img.shape[0]    
        h = img.shape[1]
        size = w * h
        if size <= 0 or tSize <= 0:
            raise IbfException('_resizeImage: Invalid image size.')     
        size_ratio = math.sqrt(float(tSize) / size)
        return cv2.resize(img, (0,0), fx=size_ratio, fy=size_ratio, interpolation = cv2.INTER_AREA) 
            
    def _getTransitionListComparingMatchedSegments(self, img1, img2, sample_cnt = 10):
    
        detector = cv2.SURF()
            
        kp1, des1 = detector.detectAndCompute(img1,None)
        kp2, des2 = detector.detectAndCompute(img2,None)
      
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
    
        good = []
        for m,n in matches:
            if m.distance < 0.50 * n.distance:
                good.append(m)
        good_match_cnt = len(good)
        
        if good_match_cnt >= sample_cnt:
            result = []              
            for i in range(0, sample_cnt):
                rannum = (random.randint(0, good_match_cnt - 1), random.randint(0, good_match_cnt - 1))
                
                while rannum[0] == rannum[1] :
                    rannum = (random.randint(0, good_match_cnt - 1), random.randint(0, good_match_cnt - 1))
                p1_idx = good[rannum[0]].queryIdx
                p1_matched_idx = good[rannum[0]].trainIdx
                p2_idx = good[rannum[1]].queryIdx
                p2_matched_idx = good[rannum[1]].trainIdx
    
                p1 = kp1[p1_idx].pt
                p1_matched = kp2[p1_matched_idx].pt
                p2 = kp1[p2_idx].pt
                p2_matched = kp2[p2_matched_idx].pt
                
                # drawing
    #            cv2.circle(img1, (int(p1[0]), int(p1[1])), 4, (255, 0, 0), 2)
    #            cv2.circle(img1, (int(p2[0]), int(p2[1])), 4, (0, 255, 0), 2)
    #            cv2.line(img1, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 1)
    #            cv2.circle(img2, (int(p1_matched[0]), int(p1_matched[1])), 4, (255, 0, 0), 2)
    #            cv2.circle(img2, (int(p2_matched[0]), int(p2_matched[1])), 4, (0, 255, 0), 2)
    #            cv2.line(img2, (int(p1_matched[0]), int(p1_matched[1])), (int(p2_matched[0]), int(p2_matched[1])), (0, 0, 255), 1)
    #            
                transition = self.__getTransitionByTwoMathcedLines(p1, p2, p1_matched, p2_matched)
                result.append(transition)
                
        else:
            err_msg = "Not enough matches are found - %d/%d" % (len(good), sample_cnt)
            raise IbfException(err_msg)
            
    #    cv2.imshow('img1', img1)
    #    cv2.imshow('img2', img2)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
        
        return result
            
    def __getTransitionByTwoMathcedLines(self, pt1, pt2, pt1_matched, pt2_matched):
        
        """
        pt1, pt2 are points(tuple type) in image1
        pt1_matched, pt2_matched are points(tuple type) in image2
        pt1_matched is a point matched to pt1
        pt2_matched is a point matched to pt2    
           
        line1 : a line constructed by pt1 and pt2 in image1
        line2 : a line constructed by pt1_matched and pt2_matched in image2
        vector of line1 : direction of this vector is from pt1 to pt2, pt1 -> pt2 (pt1 is start point and pt2 is end point)
        vector of line2 : direction of this vector is from pt1_matched to pt2_matched, pt1_matched -> pt2_patched
                            (pt1_matched is start point and pt2_matched is end point)
        """   
           
        pt1 = Vec2d(pt1)
        pt2 = Vec2d(pt2)
        pt1_matched = Vec2d(pt1_matched)
        pt2_matched = Vec2d(pt2_matched)
        
        # needs for dealing with the case of comparing same two images
        # calculate ratio of size between image1 and image2 through matched lines, line1 and line2
        line1_len = math.sqrt((pt2.x - pt1.x)**2 + (pt2.y - pt1.y)**2)
        line2_len = math.sqrt((pt2_matched.x - pt1_matched.x)**2 + (pt2_matched.y - pt1_matched.y)**2)
            
        if line1_len == 0 or line2_len == 0:
            raise IbfException('EXC, __getTransitionByTwoMatchedLines : Length of line equal zero.')
        size_ratio = round(line2_len / line1_len, 2)
    
        # caculate rotation angle from image1 to image2, using unit vector of matched lines, line1 and line2
        line1_uv = Vec2d( (pt2.x - pt1.x) / line1_len, (pt2.y - pt1.y) / line1_len )                                  # unit vector of line1
        line2_uv = Vec2d( (pt2_matched.x - pt1_matched.x) / line2_len, (pt2_matched.y - pt1_matched.y) / line2_len )  # unit vector of line2
        rot_sign = math.asin(line1_uv.x * line2_uv.y - line1_uv.y * line2_uv.x)
        if rot_sign < 0:
            rot_sign = 1
        else:
            rot_sign = -1
        #rot_angle = round(math.degrees(rot_sign * math.acos(line1_uv.x * line2_uv.x + line1_uv.y * line2_uv.y)), 0)
        rot_angle = round(rot_sign * math.acos(line1_uv.x * line2_uv.x + line1_uv.y * line2_uv.y), 2)
        if rot_angle == -0:
            rot_angle = 0
            
        pt1_resro = Vec2d(0,0)
        pt2_resro = Vec2d(pt2.x - pt1.x, pt2.y - pt1.y)
        pt2_resro.scalarProduct(size_ratio)
        pt2_resro.rotate(rot_angle)
        pt1_resro.copyObj(pt1)
        pt1_resro.scalarProduct(size_ratio)
        pt2_resro.move(pt1_resro.x, pt1_resro.y)
    
        # rigid transformation
        pt1_trans = Vec2d(round(pt1_matched.x - pt1_resro.x, 2), round(pt1_matched.y - pt1_resro.y, 2))
        
        rot_cent = pt1_resro
        
        result = {}
        result['rot_cent'] = rot_cent
        result['size_ratio'] = size_ratio
        result['rot_angle'] = rot_angle
        result['cent_trans'] = pt1_trans
         
        return result
    

def getAbs(val1, val2):
    if val1 > val2:
        return val1 - val2
    else:
        return val2 - val1


def preprocess(img):
    shape = img.shape
    rows = shape[0]
    cols = shape[1]
    for i in range(0, rows):
        for j in range(0, cols):
            img[i][j] = int(img[i][j] / 10) * 10
    return img

