import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # 1x1 Convolution Layer
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    
    # 3x3 Convolution Layer
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    
    # Another 1x1 Convolution Layer
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layer with 10 neurons
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model