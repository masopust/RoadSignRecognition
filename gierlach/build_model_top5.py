import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.metrics import top_k_categorical_accuracy

def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

def get_data():
    train_data = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,zoom_range = 0.2,rotation_range=40,fill_mode='nearest', horizontal_flip = True)
    test_data = ImageDataGenerator(rescale = 1./255)
    validation_data = ImageDataGenerator(rescale = 1./255)
    training_set = train_data.flow_from_directory('Training',target_size = (64, 64),color_mode='rgb',class_mode = 'categorical')
    test_set = test_data.flow_from_directory('Testing',target_size = (64,64),color_mode='rgb',class_mode = 'categorical')
    validation_set = validation_data.flow_from_directory('Validation',target_size = (64,64),color_mode='rgb',class_mode = 'categorical')
    plt.figure()
    plt.hist(training_set.classes, bins=62) 
    plt.title("Histogram of training dataset")
    plt.savefig('Histogram of training dataset.png')
    plt.figure()
    plt.hist(test_set.classes, bins=62)  
    plt.title("Histogram of testing dataset")
    plt.savefig("Histogram of testing dataset.png")
    plt.figure()
    plt.hist(validation_set.classes, bins=62)  
    plt.title("Histogram of validation dataset")
    plt.savefig("Histogram of validation dataset.png")

    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten(input_shape=classifier.output_shape[1:]))
    classifier.add(Dense(256, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(62, activation='softmax'))
    classifier.compile(optimizer = 'Nadam', loss = 'mean_squared_error', metrics = [top_5_accuracy])


    nr_epoch = 20
    his = classifier.fit_generator(training_set,steps_per_epoch = 110,epochs = nr_epoch,validation_data = validation_set)
    train_loss = his.history['loss']
    val_loss   = his.history['val_loss']
    train_acc  = his.history['top_5_accuracy']
    val_acc    = his.history['val_top_5_accuracy']
    xc         = range(nr_epoch)

    plt.figure()
    plt.plot(xc, val_loss, label='val_loss')
    plt.plot(xc, train_loss, label='train_loss')
    plt.ylabel("mean_squared_error")
    plt.xlabel("number of epoch")
    plt.legend()
    plt.savefig("Loss_top5.png")
    plt.figure()
    plt.plot(xc, val_acc, label='val_acc')
    plt.plot(xc ,train_acc, label='train_acc')
    plt.legend(loc=4)
    plt.ylabel("accuracy")
    plt.xlabel("number of epoch")
    plt.savefig("Accuracy_top5.png")
    classifier.save('classifier_20epoch.h5')
  

get_data()


