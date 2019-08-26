# -*- coding: utf-8 -*-

from __future__ import division

import parameters
import numpy as np
from parse import parse_size
from parse import parse_object
from onehotcode import onehotencode


def get_groundtruth(xml):
    size_dict = parse_size(xml)
    rw = 1.0*parameters.INPUT_SIZE/size_dict['width']
    rh = 1.0*parameters.INPUT_SIZE/size_dict['height']
    
    object_list = parse_object(xml)
    cell_size = int(parameters.INPUT_SIZE/parameters.S)
    
    gt = np.zeros([parameters.S,parameters.S,(5+parameters.NUM_CLASSESS)],dtype=np.float32)
    mask = np.zeros([parameters.S,parameters.S],dtype=np.float32)
    
    for box in object_list:
        box_class = box['classes']
        xmin =  box['xmin']*rw
        ymin =  box['ymin']*rh
        xmax =  box['xmax']*rw
        ymax =  box['ymax']*rh
        
        x_center = xmin + (xmax-xmin)/2.0
        y_center = ymin + (ymax-ymin)/2.0
                          
        x_cell = int(1.0*x_center/cell_size)
        y_cell = int(1.0*y_center/cell_size)
        
        x = (1.0*x_center/cell_size)-x_cell
        y = (1.0*y_center/cell_size)-y_cell
        
        sqrt_w = np.sqrt(1.0*(xmax-xmin)/(parameters.INPUT_SIZE))
        sqrt_h = np.sqrt(1.0*(ymax-ymin)/(parameters.INPUT_SIZE))
        
        ###
        # If no object exists in that cell, the confidence scores should
        # be zero. Otherwise we want the confidence score to equal the
        # intersection over union (IOU) between the predicted box and the ground truth.
        ###
        
        # If object exists in that cell, the confidence score is tentatively initialized as 1.0.
        c = 1.0    # confidence score
        bbox = [x,y,sqrt_w,sqrt_h,c]
        
        bbox = np.array(bbox,dtype=np.float32)[None]
        class_onehotcode = onehotencode([box_class+'_*'])
        
        current_gt = np.concatenate((bbox,class_onehotcode),axis=1)
        gt[y_cell,x_cell,:] = current_gt
        mask[y_cell,x_cell] = 1.0
    return {'groundtruth':gt,'mask':mask}
