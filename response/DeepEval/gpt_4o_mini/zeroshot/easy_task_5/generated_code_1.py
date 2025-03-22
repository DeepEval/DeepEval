import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.utils import to_categorical

def dl_model():
    # Input layer for 28x28 grayscale images (1 channel)
    input_shape = (28, 28, 1)
    inputs = Input(shape=input_shape)
    
    # 1x1 Convolution to reduce dimensionality
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    
    # 3x3 Convolution to extract features
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    
    # 1x1 Convolution to restore dimensionality
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layer with 10 neurons for classification
    outputs = Dense(units=10, activation='softmax')(x)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model (optional, can be done later)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example of how to use the model
if __name__ == "__main__":
    model = dl_model()
    model.summary()  # Print the model summary