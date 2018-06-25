# RoadSignRecognition

## Description
There are four different models of CNN network. We all used keras library to train our database, but we chose different approaches to build convolution network and sometimes use different data augmentation. Each model was evaluated and the best one was used to prepare program which recognise input picture and return name of the road sing.      

## Dataset
Data is taken from: [BelgiumTS Dataset](http://btsd.ethz.ch/shareddata/). There are two sets: Training and Testing. Our data-sets contains 62 class of different road signs. To get realistic learning curve, the training data was split to smaller training file and validation set. The validation set do not take part in training process. It is only used to made an evaluation during learning. The Testing data-set is used to evaluate data after training. To split data use script: [split_set.py](https://github.com/masopust/RoadSignRecognition/blob/master/gierlach/split_set.py) or use function 'split_dataset_into_test_and_train_sets', defined [here](https://colab.research.google.com/drive/1Hkcl_DUWo7M8bEhXMOydrRw5F3Np4-vP#scrollTo=uWLv9ig2ajjK). 
Below are shown histograms of numerosity of each class.

<img src="https://github.com/masopust/RoadSignRecognition/blob/master/Histogram%20of%20testing%20dataset.png" width="270" height="200" />   <img src="https://github.com/masopust/RoadSignRecognition/blob/master/Histogram%20of%20training%20dataset.png" width="270" height="200" />    <img src="https://github.com/masopust/RoadSignRecognition/blob/master/Histogram%20of%20validation%20dataset.png" width="270" height="200" />


Data was loaded using very handy method implemented in keras named flow_from_directory. It takes data from file and prepare labels for each classes based on names of subdirectories.


## Keras 
Library which helps not only to implement constitutional neural network, but also preprocess data is named: [Keras: The Python Deep Learning library](https://keras.io/). The main idea of Keras approach is to implement model in easy way as a sequence of  neural layers with all needed parameters able to set. Types of layer in CNN:
* convolutional - [documentation](https://keras.io/layers/convolutional/)
* pooling, dropout, flatten - [documentation](https://keras.io/layers/pooling/)
* dense - [documentation](https://keras.io/layers/core/)

Defining layers, which are connecting neurons, the the activation function should be pointed. [Here](https://keras.io/activations/) is a description of the available activation functions in Keras. 

When the model is designed, it must be compiled. There the [optimizer](https://keras.io/optimizers/) and statistical [metrics](https://keras.io/metrics/) should be defined.  

[Here](https://cambridgespark.com/content/tutorials/convolutional-neural-networks-with-keras/index.html) is nice tutorial about CNN concept.

Keras allows to make the image augementation, for examaple by rotating them, to increase effectiveness of learning. By setting the number of batches, the data is taken in smaller packages, so memory is seved. In that case there are to ways of setting nr of samples per epoch. First way: nr of training items/nr of batches, second way: nr_of_training_items - default. The second approach takes more time, but gives better results. 

### Something about evaluation

* Learning curve - shows how goes learning in each epoch. On y-axis are shown statistical metrics defined while compiling the model. 
* Confusion Matrix - helps notice which examples was properly classified and how many was wrong classified, indicating  false negative and false positive ones.
* Accuracy - describes difference between a result and a "true" value
* Top-n accuracy - indicates if in n numbers of results with highest probability is at least one equal to "true" value.
* Loss - error of matching, commonly used: mean squared error

## Comparison of different models
name | Masopust's model | Riva's model | Gierlach's model | Torcu's model
---|     :---:     |     :---:    |      :---:    |      :---:    
nr of epochs| 15 | 30 | 20 | x
opimizer | adam | sgd | nadam | rmsprop
diagram|<img src="https://github.com/masopust/RoadSignRecognition/blob/master/masopust_model.png" width="240" height="700" />|<img src="https://github.com/masopust/RoadSignRecognition/blob/master/riva_model.png" width="240" height="850" />|<img src="https://github.com/masopust/RoadSignRecognition/blob/master/gierlach_model.png" width="240" height="700" /> | x

## Evaluation of our models
name | Masopust's model | Riva's model | Gierlach's model | Torcu's model
---|     :---:     |     :---:    |      :---:    |      :---:  
Test accuracy | 92.6% | 96% | 92% | 
Steps per epoch | nr of items in training set | nr of items in training set| 110 | x
Learning curve Accuracy |<img src="https://github.com/masopust/RoadSignRecognition/blob/master/masopust_acc.png" width="240" height="200" />  | |<img src="https://github.com/masopust/RoadSignRecognition/blob/master/gierlach_acc.png" width="240" height="200" /> | 
Learning curve Loss|<img src="https://github.com/masopust/RoadSignRecognition/blob/master/masopust_loss.png" width="240" height="200" />  | |<img src="https://github.com/masopust/RoadSignRecognition/blob/master/gierlach_loss.png" width="240" height="200" /> | 
Learning curve Top 5 Accuracy| | | <img src="https://github.com/masopust/RoadSignRecognition/blob/master/gierlach_acc_top5.png" width="240" height="200" /> | 
### Confusion matrix
* Masopust's model
<img src="https://github.com/masopust/RoadSignRecognition/blob/master/masopust_conf.png" /> 
 
* Gierlach's model
<img src="https://github.com/masopust/RoadSignRecognition/blob/master/gierlach_conf.png" /> 

## Script to classifying the road sign - recognize.py
### Prerequisites
You need to Python 3.6, Keras and NumPy have installed. Download a model from this repository. 
### Running 
'''
python recognize.py picture_file.jpg -m model.h5
'''
## Detection of road signs 

## Authors
* **Jan Masopust** - - [masopust](https://github.com/masopust)
* **Gülce Torcu** - - [GulceTorcu](https://github.com/GulceTorcu)
* **Ewa Gierlach** - - [ewwx](https://github.com/ewwx)
* **Christian Riva** - - [chririva](https://github.com/chririva)
