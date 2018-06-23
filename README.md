# RoadSignRecognition
Python final project
## ToDo List
- [x] Choose technologies
- [x] Preprocess DB
- [x] Recognition of road sign type
- [x] Statistics
- [x] Labels
- [ ] Road sign detection

## Description
There are four different models of CNN network. We all used keras library to train our database, but we chose different approaches to build convolution network and sometimes use different data augmentation. Each model was evaluated and the best one was used to prepare program which recognise input picture and return name of the road sing.      

## Dataset
Data is taken from: [BelgiumTS Dataset](http://btsd.ethz.ch/shareddata/). There are two sets: Training and Testing. Our data-sets contains 62 class of different road signs. To get realistic learning curve, the training data was split to smaller training file and validation set. The validation set do not take part in training process. It is only used to made an evaluation during learning. The Testing data-set is used to evaluate data after training. To split data use script: [split_set.py](https://github.com/masopust/RoadSignRecognition/blob/master/gierlach/split_set.py) or use function 'split_dataset_into_test_and_train_sets', defined [here](https://colab.research.google.com/drive/1Hkcl_DUWo7M8bEhXMOydrRw5F3Np4-vP#scrollTo=uWLv9ig2ajjK). 
Below are shown histograms of numerosity of each class.

 <img src="https://github.com/masopust/RoadSignRecognition/blob/master/Histogram%20of%20testing%20dataset.png" width="270" height="200" />   <img src="https://github.com/masopust/RoadSignRecognition/blob/master/Histogram%20of%20training%20dataset.png" width="270" height="200" />    <img src="https://github.com/masopust/RoadSignRecognition/blob/master/Histogram%20of%20validation%20dataset.png" width="270" height="200" />     


## Keras 
Library which helps not only to implement constitutional neural network, but also preprocess data is named: [Keras: The Python Deep Learning library](https://keras.io/). The main idea of Keras approach is to implement model in easy way as a sequence of  neural layers with all needed parameters able to set. Types of layer in CNN:
* constitutional -- [documentation](https://keras.io/layers/convolutional/)
* pooling, dropout, flatten -- [documentation](https://keras.io/layers/pooling/)
* dense -- [documentation](https://keras.io/layers/core/)

Defining layers, which are connecting neurons, the the activation function should be pointed. [Here](https://keras.io/activations/) is a description of the available activation functions in Keras. 

When the model is designed, it must be compiled. There the [optimizer](https://keras.io/optimizers/) and statistical [metrics](https://keras.io/metrics/) should be defined.  

[Here](https://cambridgespark.com/content/tutorials/convolutional-neural-networks-with-keras/index.html) is nice tutorial about CNN concept.

### Something about evaluation
* Learning curve
* Confusion Matrix
* Accuracy

## Comparison of different models

## Evaluation of our models

## Detection of road signs 

## Authors
* **Jan Masopust** - - [masopust](https://github.com/masopust)
* **Gülce Torcu** - - [GulceTorcu](https://github.com/GulceTorcu)
* **Ewa Gierlach** - - [ewwx](https://github.com/ewwx)
* **Christian Riva** - - [chririva](https://github.com/chririva)
