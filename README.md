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
Data is taken from: [BelgiumTS Dataset](http://btsd.ethz.ch/shareddata/). There are two sets: Trainig and Testing. Our datasets contains 62 class of different road signs. To get realistic learning curve, the traing data was split to smaler traing file and validation set. The validation set do not take part in traing process. It is only used to made an evaluation during learning. The Testing dataset is used to evalaute data after trainig. To split data use script named split.py in gierlach catalog or use function 'split_dataset_into_test_and_train_sets', defined [here](https://colab.research.google.com/drive/1Hkcl_DUWo7M8bEhXMOydrRw5F3Np4-vP#scrollTo=uWLv9ig2ajjK). 
Below are shown histograms of numerousity of each class.

| :---         |     :---:      |          ---: |


## Keras 

## Comparison of different models

## Evaluation of our models

## Detection od road signs 

## Authors
* **Jan Masopust** - - [masopust](https://github.com/masopust)
* **Gülce Torcu** - - [GulceTorcu](https://github.com/GulceTorcu)
* **Ewa Gierlach** - - [ewwx](https://github.com/ewwx)
* **Christian Riva** - - [chririva](https://github.com/chririva)
