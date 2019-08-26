# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:41:35 2019

@author: LiXiaoGang
"""
from __future__ import division

import os
import datetime
import cv2 as cv
import parameters
import numpy as np
from onehotcode import onehotdecode


def calculateIoU(xmin0,ymin0,xmax0,ymax0,xmin1,ymin1,xmax1,ymax1):
    '''
    计算两个边界框的iou
    注：传递入坐标为边界框的左上角和右下角坐标，并且已经被输入图像的宽、高归一化至0~1之间
    '''
    w = max(0.0, min(xmax0, xmax1) - max(xmin0, xmin1))
    h = max(0.0, min(ymax0, ymax1) - max(ymin0, ymin1))
    intersection = w*h
    union = (xmax0-xmin0)*(ymax0-ymin0)+(xmax1-xmin1)*(ymax1-ymin1)-intersection
    
    if union<=0.0:
        iou = 0.0
    else:
        iou = 1.0*intersection / union
    return iou


def box_decode(predictions,imgname,threshold):
    '''
    将预测结果还原成边界框的左上角和右下角坐标，并计算类别置信度分数
    '''
    grid_s = predictions.shape[0]
    
    boxes = []
    for sh in range(grid_s):
        for sw in range(grid_s):
            one_cell_pred = predictions[sh,sw,0::]
            for b in range(parameters.B):
                box = one_cell_pred[(5*b):(5*b+5)]
                if box[-1] >= threshold:
                    xmin = max(0.0,((sw+box[0])/grid_s - (box[2]**2)/2.0))
                    ymin = max(0.0,((sh+box[1])/grid_s - (box[3]**2)/2.0))
                    xmax = min(1.0,((sw+box[0])/grid_s + (box[2]**2)/2.0))
                    ymax = min(1.0,((sh+box[1])/grid_s + (box[3]**2)/2.0))
                           
                    prob = box[-1]*one_cell_pred[-parameters.NUM_CLASSESS::]
                    pred_class = onehotdecode(prob)
                    max_prob = max(prob)
                           
                    pred_box = {'box':[xmin,ymin,xmax,ymax,max_prob],'className':pred_class}
                    boxes.append(pred_box)
    result = {'imageName':imgname,'boxes':boxes}
    return result


def nms(result,threshold):
    '''
    使用非极大值抑制算法(Non-maximal suppression)去除检测出来的冗余边界框
    '''
    class_list =[]
    final_pred_boxes = []
    boxes = result['boxes']
    
    for b in range(len(boxes)):
        class_list.append(boxes[b]['className'])
    class_list = np.unique(class_list)
    
    for name in class_list:
        box_coord = []
        for b in range(len(boxes)):
            if name == boxes[b]['className']:
                box_coord.append(boxes[b]['box'])       
        box_coord = np.array(box_coord)
        
        while box_coord.shape[0] > 0:                
            idx = np.argmax(box_coord[:,-1])
            keep_box = box_coord[idx,:]
            pred_box = {'box':keep_box,'className':name}
            final_pred_boxes.append(pred_box)
            
            box_coord = np.delete(box_coord,[idx],axis=0)
            if box_coord.shape[0] == 0:break
            
            suppre = []
            xmin0 = keep_box[0]
            ymin0 = keep_box[1]
            xmax0 = keep_box[2]
            ymax0 = keep_box[3]
            
            for b in range(box_coord.shape[0]):
                xmin1 = box_coord[b,:][0]
                ymin1 = box_coord[b,:][1]
                xmax1 = box_coord[b,:][2]
                ymax1 = box_coord[b,:][3]
                
                iou = calculateIoU(xmin0,ymin0,xmax0,ymax0,
                                   xmin1,ymin1,xmax1,ymax1)
                if iou > threshold:
                    suppre.append(b)
            box_coord = np.delete(box_coord,suppre,axis=0)
    detections = {'imageName':result['imageName'],'boxes':final_pred_boxes}
    return detections


def save_instance(detections):
    image_name = detections['imageName'][0]+'.'+parameters.pic_type
    read_dir = os.path.join(parameters.path,'data','annotation','images',image_name)
    write_dir = os.path.join(parameters.path,'pic')
    
    im = cv.imread(read_dir).astype(np.float32)
    im_h = im.shape[0]
    im_w = im.shape[1]
    
    im = cv.resize(im,(parameters.INPUT_SIZE,parameters.INPUT_SIZE)).astype(np.float32)
    for b in range(len(detections['boxes'])):
        box = detections['boxes'][b]['box']
        name = detections['boxes'][b]['className']
        
        xmin = int(box[0]*parameters.INPUT_SIZE)
        ymin = int(box[1]*parameters.INPUT_SIZE)
        xmax = int(box[2]*parameters.INPUT_SIZE)
        ymax = int(box[3]*parameters.INPUT_SIZE)
        prob = round(box[4]*100)
        txt = name +':'+ str(prob) + '%'
        
        font = cv.FONT_HERSHEY_PLAIN
        im = cv.rectangle(im,(xmin,ymin),(xmax,ymax),(255, 0, 0),1)
        im = cv.putText(im,txt,(xmin,ymin),font,1,(255,0,0),1)
    
    im = cv.resize(im,(im_w,im_h)).astype(np.float32)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-')
    dst = os.path.join(write_dir,current_time+image_name)
    cv.imwrite(dst,im)