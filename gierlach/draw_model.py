from keras.models import load_model
from keras.utils import plot_model

model = load_model('classifier_augmentation1_20epoch.h5')
plot_model(model, to_file='model.png')





