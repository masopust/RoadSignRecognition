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