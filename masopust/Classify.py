# Importing the Keras libraries and packages
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model

def main():
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    training_set = train_datagen.flow_from_directory('training', target_size=(64, 64), batch_size=32, class_mode='categorical')
    test_set = test_datagen.flow_from_directory('testing', target_size=(64, 64), batch_size=32, class_mode='categorical')

    print(len(training_set.filenames))
    indices = training_set.class_indices
    print(indices)
    print(len(training_set.class_indices))


    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten(input_shape=classifier.output_shape[1:]))
    # Step 4 - Full connection
    classifier.add(Dense(units = 256, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 62, activation = 'softmax'))
    # Compiling the CNN
    classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    print(type(training_set[0][1]))
    print(len(training_set[0][1]))
    print(training_set[0][1][0])


    # Part 2 - Fitting the CNN to the images
    classifier.fit_generator(training_set,steps_per_epoch = 4575, epochs = 25, validation_data = test_set, validation_steps = 2520)

    classifier.save('my_classifier.h5')
    model = load_model('my_classifier.h5')
    # Part 3 - Making new predictions

    test_image = image.load_img('single_prediction/abc.jpg', target_size = (64, 64))
    #test_image = test_image / 255
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    print(result)


if __name__ == '__main__':
    #freeze_support()
    main()