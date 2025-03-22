# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
import numpy as np
import matplotlib.pyplot as plt


# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class labels to categorical labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Define the first convolutional layer with 1x1 kernel
    x = layers.Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    
    # Define the first max pooling layer with 3x3 kernel
    x = layers.MaxPooling2D((3, 3))(x)
    
    # Define the second convolutional layer with 3x3 kernel
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    
    # Define the second max pooling layer with 3x3 kernel
    x = layers.MaxPooling2D((3, 3))(x)
    
    # Define the third convolutional layer with 5x5 kernel
    x = layers.Conv2D(128, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    
    # Define the third max pooling layer with 3x3 kernel
    x = layers.MaxPooling2D((3, 3))(x)
    
    # Define the global average pooling layer
    x = layers.GlobalAveragePooling2D()(x)
    
    # Define the first fully connected layer
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    
    # Define the second fully connected layer
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# Define the model
model = dl_model()

# Define the data augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Train the model
history = model.fit(data_augmentation.flow(x_train, y_train, batch_size=32), epochs=10, validation_data=(x_test, y_test))