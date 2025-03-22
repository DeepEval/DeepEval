import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, ReLU, Conv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def block(input_tensor):
    # Batch normalization and ReLU activation
    x = BatchNormalization()(input_tensor)
    x = ReLU()(x)
    
    # 3x3 Convolution layer
    x = Conv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), padding='same')(x)
    
    # Concatenation with the input tensor
    x = Concatenate(axis=-1)([input_tensor, x])
    
    return x

def pathway(input_tensor):
    x = block(input_tensor)
    x = block(x)
    x = block(x)
    return x

def dl_model():
    # Define the input
    input_tensor = Input(shape=(28, 28, 1))  # MNIST images are 28x28 with 1 channel

    # Two processing pathways
    pathway1 = pathway(input_tensor)
    pathway2 = pathway(input_tensor)

    # Concatenate outputs from both pathways
    merged = Concatenate(axis=-1)([pathway1, pathway2])

    # Flatten and add fully connected layers
    x = Flatten()(merged)
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Create model
    model = Model(inputs=input_tensor, outputs=output)

    return model