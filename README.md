# Object Detection
This is the implementation of Video Transformer Network (VTN) approach for Action Recognition in Tensorflow. It contains complete code for preprocessing,training and test. Besides, this repository is easy-to-use and can be developed on Linux and Windows.  

[VTN : Kozlov, Alexander, Vadim Andronov, and Yana Gritsenko. "Lightweight Network Architecture for Real-Time Action Recognition." arXiv preprint arXiv:1905.08711 (2019).](https://arxiv.org/abs/1905.08711)

## Getting Started
### 1 Prerequisites  
* Python3.6  
* Tensorflow  
* Opencv-python  
* Pandas  

### 2 Download this repo and unzip it  
`cd ../VTN/Label_Map`  
Open the `label.txt` and revise its class names as yours.  

### 3 Generate directory  
`cd ../VTN/Code`  
`run python make_dir.py`  
Then some subfolders will be generated in  `../VTN/Raw_Data` , `../VTN/Data/Train`,  `../VTN/Data/Test`, `../VTN/Data/Val`, where name of the subfolders is your class names defined in `label.txt`.  

### 4 Prepare video clips  
According to the class, copy your raw AVI videos to subfolders in `../VTN/Raw_Data`. Optionally, you can use the public HMDB-51 dataset, which can be found [here](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).  
`cd ../VTN/Code`  
`run python prepare_clips.py`  
Clips generated will be saved in the subfolders in   `../VTN/Data/Train`,  `../VTN/Data/Test`, `../VTN/Data/Val`. These clips will be used for training, test and validation.  

### 5 Compute the mean image from training clips(optional)  
`cd ../VTN/Code`  
`run python mean_img.py`    
And then a mean image is saved in directory `../VTN/Data/Train`.  

### 6 Train model  
The model parameters, training parameters and eval parameters are all defined by `parameters.py`.  
`cd ../VTN/Code`  
`run python train.py PB` or `python train.py CHECKPOINT`  
The model will be saved in directory `../VTN/Model`, where "PB" and "CHECKPOINT" is two ways used for saving model for Tensorflow.  
 
### 7 Test model(pb)  
Test model using clips in `../VTN/Data/Test`.  
`cd ../VTN/Code`  
`run python test.py N`  
Where N is not more than the number of clips in test set. Note that we do not use batch during test. There may be out of memory errors with a large N. In this case, you can modify the `test.py` to use batch.    

### 8 Visualize model using Tensorboard  
`cd ../VTN`  
`run tensorboard --logdir=Model/`   
Open the URL in browser to visualize model.  
