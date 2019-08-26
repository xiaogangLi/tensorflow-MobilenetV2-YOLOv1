# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:46:14 2019
https://github.com/xiongzihua/pytorch-YOLO-v1
https://github.com/lovekittynine/my_tensorflow_yolo
@author: LiXiaoGang        
"""
from __future__ import division

import os
import shutil
import parameters
import numpy as np
import tensorflow as tf
from readbatch import mini_batch
from mobilenetv2 import mobilenetv2
from postprocessing import nms,box_decode,save_instance



def net_placeholder(batch_size=None):
    Input = tf.placeholder(dtype=tf.float32,
                            shape=[batch_size,
                            parameters.INPUT_SIZE,
                            parameters.INPUT_SIZE,
                            parameters.CHANNEL],name='Input')
    
    Label = tf.placeholder(dtype=tf.float32,
                            shape=[batch_size,
                            parameters.S,
                            parameters.S,
                            (5+parameters.NUM_CLASSESS)],name='Label')
    
    Mask = tf.placeholder(dtype=tf.float32,
                          shape=[batch_size,
                          parameters.S,
                          parameters.S],name='Mask')
    
    isTraining = tf.placeholder(tf.bool,name='Batch_norm')
    return Input,Label,Mask,isTraining


def isobject(Label,logits,i,j,k):
    # x,y,sqrt_w,sqrt_h,c
    bbox_gt = Label[i,j,k,0:5]
    bbox_gt_xmin = (k+bbox_gt[0])/parameters.S - (bbox_gt[2]**2)/2.0
    bbox_gt_ymin = (j+bbox_gt[1])/parameters.S - (bbox_gt[3]**2)/2.0
    bbox_gt_xmax = (k+bbox_gt[0])/parameters.S + (bbox_gt[2]**2)/2.0
    bbox_gt_ymax = (j+bbox_gt[1])/parameters.S + (bbox_gt[3]**2)/2.0   

    max_iou = 0.0
    loss_collection = []
    bbox_pred_highest_iou = tf.zeros(shape=bbox_gt.get_shape(),dtype=tf.float32)
    for b in range(parameters.B):
        
        bbox_pred = logits[i,j,k,(5*b):(5*b+5)]
        bbox_pred_xmin = (k+bbox_pred[0])/parameters.S - (bbox_pred[2]**2)/2.0
        bbox_pred_ymin = (j+bbox_pred[1])/parameters.S - (bbox_pred[3]**2)/2.0
        bbox_pred_xmax = (k+bbox_pred[0])/parameters.S + (bbox_pred[2]**2)/2.0
        bbox_pred_ymax = (j+bbox_pred[1])/parameters.S + (bbox_pred[3]**2)/2.0
         
        # iou         
        w = tf.maximum(0.0, tf.minimum(bbox_gt_xmax, bbox_pred_xmax) - tf.maximum(bbox_gt_xmin, bbox_pred_xmin))
        h = tf.maximum(0.0, tf.minimum(bbox_gt_ymax, bbox_pred_ymax) - tf.maximum(bbox_gt_ymin, bbox_pred_ymin))
        intersection = w*h
        union = (bbox_gt_xmax-bbox_gt_xmin)*(bbox_gt_ymax-bbox_gt_ymin)+(bbox_pred_xmax-bbox_pred_xmin)*(bbox_pred_ymax-bbox_pred_ymin)-intersection
        iou = tf.cond(union<=0.0,lambda:tf.cast(0.0,tf.float32),lambda:intersection/union)
        
        bbox_pred_highest_iou = tf.cond(max_iou<=iou,lambda:bbox_pred,lambda:bbox_pred_highest_iou)
        max_iou = tf.cond(max_iou<=iou,lambda:iou,lambda:max_iou)
        
    loss_collection.append(tf.reduce_sum(parameters.LAMBDA_COORD*tf.square(tf.subtract(bbox_gt[0:4],bbox_pred_highest_iou[0:4]))))
    loss_collection.append(tf.reduce_sum(tf.square(tf.subtract(max_iou,bbox_pred_highest_iou[4]))))
    loss_collection.append(tf.reduce_sum(tf.square(tf.subtract(Label[i,j,k,-parameters.NUM_CLASSESS::],logits[i,j,k,-parameters.NUM_CLASSESS::]))))
    loss_collection = tf.reduce_sum(loss_collection)
    return loss_collection
    
    
def noobject(Label,logits,i,j,k):
    # x,y,sqrt_w,sqrt_h,c
    loss_collection = []
    c_gt = Label[i,j,k,4]    # Label[i,j,k,4] always is 0.
    
    for b in range(parameters.B):
        c_pred = logits[i,j,k,5*(b+1)-1]
        loss_collection.append(parameters.LAMBDA_NOOBJ*tf.square(tf.subtract(c_gt,c_pred)))
    loss_collection = tf.reduce_sum(loss_collection)
    return loss_collection
    
    
def net_loss(Label,mask,logits):
    loss = []
    for i in range(parameters.BATCH_SIZE):
        loss_collection = []
        for j in range(parameters.S):
            for k in range(parameters.S):
                choice = tf.equal(mask[i,j,k],tf.constant(1.0,dtype=tf.float32))
                loss_cell = tf.cond(choice,lambda:isobject(Label,logits,i,j,k),lambda:noobject(Label,logits,i,j,k))
                loss_collection.append(loss_cell)
        loss.append(tf.reduce_sum(loss_collection))
    loss = tf.reduce_mean(loss,name='Loss')
    return loss


def training_net():
    image,label,mask,isTraining = net_placeholder(None)
    logits = mobilenetv2(image,isTraining)
    loss = net_loss(label,mask,logits)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):        
        train_step = tf.train.AdamOptimizer(parameters.LEARNING_RATE).minimize(loss)
     
    Saver = tf.train.Saver(var_list=tf.global_variables(),max_to_keep=5)
    with tf.Session() as sess:
        
        writer = tf.summary.FileWriter(os.path.join(parameters.path,'model'), sess.graph)     
        init_var_op = tf.global_variables_initializer()
        sess.run(init_var_op)
        
        for i in range(parameters.TRAIN_STEPS):
            batch = mini_batch(i,parameters.BATCH_SIZE,'train')
            feed_dict = {image:batch['image'], label:batch['label'], mask:batch['mask'], isTraining:True}
            _,loss_ = sess.run([train_step,loss],feed_dict=feed_dict)
            print('===>Step %d: loss = %g ' % (i,loss_))
            
            # evaluate and save checkpoint
            if i % 500 == 0:
                write_instance_dir = os.path.join(parameters.path,'pic')
                if not os.path.exists(write_instance_dir):os.mkdir(write_instance_dir)
                j = 0
                while True:
                    
                    batch = mini_batch(j,1,'val')
                    feed_dict = {image:batch['image'],isTraining:False}
                    pred_output = sess.run(logits,feed_dict=feed_dict)
    
                    pred_output = np.squeeze(pred_output)
                    pred_output = box_decode(pred_output,batch['image_name'],parameters.SCORE_THRESHOLD)
                    pred_output = nms(pred_output,parameters.NMS_THRESHOLD)
                    
                    if j < min(20,batch['image_num']):save_instance(pred_output)
                    if j == batch['image_num']-1:break
                    j += 1
                
                if os.path.exists(parameters.CHECKPOINT_MODEL_SAVE_PATH) and (i==0):
                    shutil.rmtree(parameters.CHECKPOINT_MODEL_SAVE_PATH)
                Saver.save(sess,os.path.join(parameters.CHECKPOINT_MODEL_SAVE_PATH,parameters.MODEL_NAME))             
            

def main():
    training_net()
     
if __name__ == '__main__':
    main()