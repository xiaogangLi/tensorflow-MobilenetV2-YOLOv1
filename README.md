# Object Detection
This is the implementation of YOLOv1 for object detection in Tensorflow. It contains complete code for preprocessing, training and test. Besides, this repository is easy-to-use and can be developed on Linux and Windows.  

[YOLOv1 : Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.](https://arxiv.org/abs/1506.02640)

## Getting Started
### 1 Prerequisites  
* Python3.6  
* Tensorflow  
* Opencv-python  
* Pandas  

### 2 Define your class names  
Download  and unzip this repository.  
`cd ../YOLOv1/label`  
Open the `label.txt` and revise its class names as yours.  

### 3 Prepare images  
Copy your images and annotation files to directories `../YOLOv1/data/annotation/images` and `../YOLOv1/data/annotation/images/xml` separately, where the annotations should be obtained by [a graphical image annotation tool](https://github.com/tzutalin/labelImg) and  saved as XML files in PASCAL VOC format.  
`cd ../YOLOv1/Code`  
`run python spilt.py`  
Then train and val images will be generated in  `../YOLOv1/data/annotation/train` and  `/YOLOv1/data/annotation/test` directories, separately.  

### 4 Train model  
The model parameters, training parameters and eval parameters are all defined by `parameters.py`.  
`cd ../YOLOv1/Code`  
`run python train.py`  
The model will be saved in directory `../YOLOv1/model/checkpoint`, and some detection results are saved in `../YOLOv1/pic`. 
 
### 5 Visualize model using Tensorboard  
`cd ../YOLOv1`  
`run tensorboard --logdir=model/`   
Open the URL in browser to visualize model.  

![image](https://github.com/xiaogangLi/tensorflow-MobilenetV2-YOLOv1/blob/master/YOLOv1/pic/example.jpg)
