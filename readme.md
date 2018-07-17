# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

the original project is:
https://github.com/udacity/CarND-Behavioral-Cloning-P3

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/2.png "On the fork of road"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the center lane driving. 

Note that I trained the model on my local host, with a NVIDIA 940M. Compared to the given dataset, additional driving records (about 1000 images) were included, to ensure the vehical stay on the track when it on the fork in a road.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the Lenet.

My first step was to use a convolution neural network model, with 3 conv-layers and 3 dense layers. Since it is a regression model, one output node was adopted. I thought this model might be appropriate because larger neural networks exhaust the memory, and can not be trained both on the service and on my own computer with 16GiB memory.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a similar mean squared error on the training set and on the validation set. This implied that the model was not overfitting. 

I think the model was not overfitted because the model is small. It was rather underfitted but not overfitted. 


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to the fork of the road. To improve the driving behavior in these cases, some additional data (about 1K images) for the training set were used.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers :

i  : Lambda Layer;
ii : 3 conv-layers
      conv-relu-pooling->
      conv-relu-pooling->
      conv-relu-pooling->
iii: Dropout layer
iv : Flattern layer
v  : 3 Dense layers
     dense-layer-relu->
     dense-layer-relu->
     dense-layer(one output node)


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving.


After the collection process, I had 10K number of data points. 


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by 7. I used an adam optimizer so that manually training the learning rate wasn't necessary.
