from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# Convert the labels to categorical one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the input shape
input_shape = (28, 28, 1)

# Define the model
model = Sequential()
model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))