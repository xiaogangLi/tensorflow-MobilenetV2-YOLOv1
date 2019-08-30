# -*- coding: utf-8 -*-

from __future__ import division

import os
import pandas as pd


path = os.path.dirname(os.getcwd())
labels = pd.read_csv(os.path.join(path,'label','label.txt'))

# training para
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
TRAIN_STEPS = 150000

ratio = 0.25
pic_type = 'jpg'    # the picture format of training images.

# yolo para
S = 7    # divides the input image into an S x S grid.
B = 2    # predicts B bounding boxes in each grid cell.
CHANNEL = 3
INPUT_SIZE = 448
NUM_CLASSESS = len(labels.Class_name)
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5
SCORE_THRESHOLD = 0.1    # background when confidence scores < 0.1
NMS_THRESHOLD = 0.5

MODEL_NAME = 'model.ckpt'
CHECKPOINT_MODEL_SAVE_PATH = os.path.join(path,'model','checkpoint')
