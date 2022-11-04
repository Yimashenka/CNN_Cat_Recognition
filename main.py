#Convolutional Neural Network

## Step 1 : Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

## Step 2 : Data Preprocessing
### Preprocessing the training set
'''
In order to prevent the overfitting, I will apply some transformations to the
pictures in the training set (rotation, zoom, flips, etc). This is called ima-
ge augmentation, using the keras method :
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
'''

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    'data/training_set/',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

### Preprocessing the test set
test_datagen = ImageDataGenerator(
    rescale=1./255
)
test_set = test_datagen.flow_from_directory(
    'data/test_set/',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)