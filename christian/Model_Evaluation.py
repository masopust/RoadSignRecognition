import numpy as np
from skimage import io
import os
from keras import backend as K
K.set_image_data_format('channels_first')
import pandas as pd
from keras.models import load_model
from SignRecognition import preprocess_img
import glob

#test = pd.read_csv('DATASET\\GT-final_test.csv', sep=';')
model = load_model('model.h5')

# Load test dataset
X_test = []
y_test = []
i = 0
'''
for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
    img_path = os.path.join('GTSRB/Final_Test/Images/', file_name)
    X_test.append(preprocess_img(io.imread(img_path)))
    y_test.append(class_id)
'''
root_dir = 'DATASET\\Testing\\'
all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    X_test.append(preprocess_img(io.imread(img_path)))
    y_test.append(int(img_path.split('\\')[-2]))

X_test = np.array(X_test)
y_test = np.array(y_test)

# predict and evaluate
y_pred = model.predict_classes(X_test)
acc = np.sum(y_pred == y_test) / np.size(y_pred)
print("Test accuracy = {}".format(acc))