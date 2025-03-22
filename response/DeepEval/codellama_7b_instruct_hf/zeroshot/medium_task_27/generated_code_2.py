import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os

# Functional API of Keras
def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Convert class labels to binary vectors
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the model
model = dl_model()