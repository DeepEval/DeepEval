# Import necessary packages
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
from tensorflow import keras
import numpy as np

def dl_model():
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    
    The model starts with two sequential layers comprising a convolutional layer followed by a max pooling layer, 
    followed by an additional convolutional layer. After these layers, the feature maps are flattened into a one-dimensional vector. 
    This vector is then processed by two fully connected layers to produce the final classification results.
    
    Parameters:
    None
    
    Returns:
    A compiled Keras model.
    """
    
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Preprocess the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # Construct the model
    inputs = layers.Input(shape=(28, 28, 1))
    
    # Convolutional and max pooling layer 1
    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Convolutional layer 2
    conv2 = layers.Conv2D(64, (3, 3), activation='relu')(pool1)
    
    # Flatten the feature maps
    flatten = layers.Flatten()(conv2)
    
    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flatten)
    outputs = layers.Dense(10, activation='softmax')(dense1)
    
    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Test the model
model = dl_model()
print(model.summary())