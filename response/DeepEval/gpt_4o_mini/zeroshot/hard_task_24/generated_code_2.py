import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dense, Flatten
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 image size
    inputs = Input(shape=input_shape)
    
    # 1x1 initial convolutional layer
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    
    # Branch 1: Extracts local features through a 3x3 convolutional layer
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Branch 2: Max pooling, 3x3 convolutional, upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2))(x)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)
    
    # Branch 3: Max pooling, 3x3 convolutional, upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2))(x)
    branch3 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)
    
    # Concatenating the outputs of all branches
    merged = concatenate([branch1, branch2, branch3])
    
    # 1x1 convolutional layer after concatenation
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(merged)

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example of creating the model
model = dl_model()
model.summary()  # Display the model's architecture