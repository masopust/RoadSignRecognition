import numpy as np
import argparse
import sys
from keras.preprocessing import image
from keras.models import load_model

def main():

    parser = argparse.ArgumentParser(
        description='Road sign recognition (classification) based on model created e.g. by Keras library.')
    parser.add_argument("-m", "--model", type=str, nargs='?', const='classifier.h5', default='classifier.h5', help="the name of the h5 file with the model")
    parser.add_argument("file", type=str, help="the name of the image file with road sign")
    args = parser.parse_args()

    model = load_model(args.model)
    labels = ["Uneven road", "Speed bump", "Slippery road", "Dangerous curve to the left",
              "Dangerous curve to the right", "Double dangerous curve to the left",
              "Double dangerous curve to the right", "Presence of children", "Bicycle crossing", "Cattle crossing",
              "Road works ahead", "Traffic signals ahead", "Guarded railroad crossing", "Indefinite danger",
              "Road narrows", "Road narrows from the left", "Road narrows from the right",
              "Priority at the next intersection", "Intersection where the priority from the right is applicable",
              "Yield right of way", "Yield to oncoming traffic", "Stop", "No entry for all drivers",
              "No bicycles allowed", "Maximum weights allowed (including load)", "No cargo vehicles allowed",
              "Maximum width allowed", "Maximum height allowed", "No traffic allowed in both directions",
              "No left turn", "No right turn",
              "No passing to the left vehicles having more than 2 wheels and horse drawn vehicles",
              "Maximum speed limit", "Mandatory way for pedestrians and bicycles", "Mandatory direction (straight on)",
              "Mandatory direction (to the right or to the left)", "Mandatory directions(straight on and to the right)",
              "Mandatory traffic circle", "Mandatory bicycle path",
              "Path shared between pedestrians, bicycles and mopeds class A", "No parking", "No waiting or parking",
              "No parking from the 1 st to the 15th of the month", "No parking from the 16th till the end of the month",
              "Priority over oncoming traffic", "Parking allowed", "Additional parking sign for handicap only",
              "Parking exclusively for motorcars", "Parking exclusively for trucks",
              "Parking exclusively for buses/coaches", "Parking on sidewalk or verge mandatory",
              "Beginning of a residential area", "End of a residential area", "One way traffic", "Dead end",
              "End of road works", "Pedestrian crosswalk", "Bicycles and mopeds crossing", "Parking ahead",
              "Speed bump", "End of priority road", "Priority road"]
    test_image = image.load_img(args.file, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    y_pred = np.argmax(result, axis=-1)
    print(labels[int(y_pred)])

if __name__ == '__main__':
    main()