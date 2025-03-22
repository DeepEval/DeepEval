import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, ReLU, Flatten, Dense
from keras.layers import Layer
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np

def dl_model():
    # Input shape should match the MNIST dataset
    input_shape = (28, 28, 1)  # Adjust based on the actual MNIST image size
    num_classes = 10  # Since it's a classification task on MNIST

    # Pathways
    pathway1 = create_pathway(input_shape)
    pathway2 = create_pathway(input_shape)

    # Concatenate outputs from both pathways
    merged = Concatenate()([pathway1.output, pathway2.output])

    # Classification using two fully connected layers
    output = Dense(128, activation='relu')(merged)
    output = Dense(num_classes, activation='softmax')(output)

    # Model
    model = Model(inputs=[pathway1.input, pathway2.input], outputs=output)

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    return model

def create_pathway(input_shape):
    input_layer = Input(shape=input_shape)
    
    # Block structure (repeated three times)
    for _ in range(3):
        input_layer = Conv2D(32, kernel_size=3, padding='same', activation=ReLU(),
                             kernel_initializer='he_normal')(input_layer)
        input_layer = BatchNormalization()(input_layer)
        input_layer = MaxPooling2D(pool_size=2, strides=2, padding='same')(input_layer)
    
    # Flatten and fully connected layer
    flat = Flatten()(input_layer)
    dense = Dense(128, activation='relu')(flat)
    
    return Model(inputs=input_layer, outputs=dense)

# Instantiate and return the model
model = dl_model()
model.summary()