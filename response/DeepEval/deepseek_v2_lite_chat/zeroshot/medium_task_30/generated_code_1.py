from tensorflow import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.layers import AveragePooling2D


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

input_shape = x_train[0].shape


def dl_model():
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Three average pooling layers with different window sizes and strides
    p1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    p2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(p1)
    p3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(p2)
    
    # Flatten the outputs and concatenate them
    flat1 = Flatten()(p3)
    flat2 = Flatten()(p2)
    concat = concatenate([flat1, flat2])
    
    # Fully connected layers
    dense1 = Dense(128, activation='relu')(concat)
    dense2 = Dense(64, activation='relu')(dense1)
    
    # Output layer
    output = Dense(10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Build the model
model = dl_model()

# Print the model summary
model.summary()