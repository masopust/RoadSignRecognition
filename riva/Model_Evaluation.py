import numpy as np
from skimage import io
import os
from keras import backend as K
K.set_image_data_format('channels_first')
import pandas as pd
from ConfusionMatrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from SignRecognition import preprocess_img
import glob

#test = pd.read_csv('DATASET\\GT-final_test.csv', sep=';')
model = load_model('model.h5')
labels = ["Uneven road", "Speed bump", "Slippery road", "Dangerous curve to the left", "Dangerous curve to the right", "Double dangerous curve to the left", "Double dangerous curve to the right", "Presence of children", "Bicycle crossing", "Cattle crossing", "Road works ahead", "Traffic signals ahead", "Guarded railroad crossing", "Indefinite danger", "Road narrows", "Road narrows from the left", "Road narrows from the right", "Priority at the next intersection", "Intersection where the priority from the right is applicable", "Yield right of way", "Yield to oncoming traffic", "Stop", "No entry for all drivers", "No bicycles allowed", "Maximum weights allowed (including load)", "No cargo vehicles allowed", "Maximum width allowed", "Maximum height allowed", "No traffic allowed in both directions", "No left turn", "No right turn", "No passing to the left vehicles having more than 2 wheels and horse drawn vehicles", "Maximum speed limit", "Mandatory way for pedestrians and bicycles", "Mandatory direction (straight on)", "Mandatory direction (to the right or to the left)", "Mandatory directions(straight on and to the right)", "Mandatory traffic circle", "Mandatory bicycle path", "Path shared between pedestrians, bicycles and mopeds class A", "No parking", "No waiting or parking", "No parking from the 1 st to the 15th of the month", "No parking from the 16th till the end of the month", "Priority over oncoming traffic", "Parking allowed", "Additional parking sign for handicap only", "Parking exclusively for motorcars", "Parking exclusively for trucks", "Parking exclusively for buses/coaches", "Parking on sidewalk or verge mandatory", "Beginning of a residential area", "End of a residential area", "One way traffic", "Dead end", "End of road works", "Pedestrian crosswalk", "Bicycles and mopeds crossing", "Parking ahead", "Speed bump", "End of priority road", "Priority road"]
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

cnf_matrix = confusion_matrix(y_test, model.predict(X_test, verbose = 1))
plot_confusion_matrix(cnf_matrix, list(range(0, 62)), True)
