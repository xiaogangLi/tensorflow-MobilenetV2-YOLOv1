# -*- coding: utf-8 -*-

import os
import sys
import cv2 as cv
import parameters
import numpy as np
import pandas as pd
from groundtruth import get_groundtruth


train_image_name = pd.read_csv(os.path.join(parameters.path,'data','train','train.txt'),header=None,names=['Name'])
val_image_name = pd.read_csv(os.path.join(parameters.path,'data','val','val.txt'),header=None,names=['Name'])

def mini_batch(i,batch_size,flag):
    if flag == 'train':
        data_name = train_image_name
        
    elif flag == 'val':
        data_name = val_image_name
        
    else:
        print('The argument "%s"  does not exist!' % (flag))
        sys.exit(0)
        
    start = (i*batch_size) % len(data_name['Name'])
    end = min(start+batch_size,len(data_name['Name']))
    
    if (end - start) < batch_size:
        start = len(data_name['Name']) - batch_size
        end = len(data_name['Name'])
    
    image = np.zeros([batch_size,parameters.INPUT_SIZE,parameters.INPUT_SIZE,parameters.CHANNEL],dtype=np.float32)
    label = np.zeros([batch_size,parameters.S,parameters.S,(5+parameters.NUM_CLASSESS)],dtype=np.float32)
    mask = np.zeros([batch_size,parameters.S,parameters.S],dtype=np.float32)
        
    batch_name = np.array(data_name['Name'][start:end])
    for j in range(len(batch_name)):
        image_name = os.path.join(parameters.path,'data','annotation','images',batch_name[j]+'.'+parameters.pic_type)
        im = cv.imread(image_name).astype(np.float32)
        image[j,:,:,:] = cv.resize(im,(parameters.INPUT_SIZE,parameters.INPUT_SIZE)).astype(np.float32)
        
        xml_name = os.path.join(parameters.path,'data','annotation','xml',batch_name[j]+'.xml')
        groundtruth = get_groundtruth(xml_name)
        label[j,:,:,:] = groundtruth['groundtruth']
        mask[j,:,:] = groundtruth['mask']
    return {'image':image,'label':label,'mask':mask,'image_name':batch_name,'image_num':data_name.shape[0]}
