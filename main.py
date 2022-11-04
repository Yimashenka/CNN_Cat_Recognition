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

## Step 3 : Building the CNN
### Initialising the CNN
cnn = tf.keras.models.Sequential()

### 1 - Convolution :
cnn.add(
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        input_shape=[64, 64, 3]
    )
)

### 2 - Pooling :
cnn.add(
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=2
    )
)

### Adding a second Convolutional layer
cnn.add(
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        input_shape=[64, 64, 3]
    )
)

cnn.add(
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=2
    )
)

### 3 - Flattening :
cnn.add(
    tf.keras.layers.Flatten()
)

### 4 - Full Connection :
cnn.add(
    tf.keras.Dense(
        units=128,
        activation='relu'
    )
)

### 5 - Output layer :
cnn.add(
    tf.keras.Dense(
        units=1,
        activation='sigmoid'
    )
)

## Step 4 : Training the CNN
### Compiling the CNN
cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

### Training the CNN
cnn.fit(
    x=training_set,
    validation_data=test_set,
    epochs=25
)

