import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolutional layer
    x = Conv2D(32, (1, 1), activation='relu')(input_layer)
    
    # Branch 1: Extract local features through a 3x3 convolutional layer
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Branch 2: Sequential layers for downsampling and feature extraction
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    branch2 = MaxPooling2D((2, 2))(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = UpSampling2D((2, 2))(branch2)
    
    # Branch 3: Sequential layers for downsampling and feature extraction
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    branch3 = MaxPooling2D((2, 2))(branch3)
    branch3 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = UpSampling2D((2, 2))(branch3)
    
    # Concatenate the outputs of all branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Pass through another 1x1 convolutional layer
    x = Conv2D(64, (1, 1), activation='relu')(concatenated)
    
    # Flatten the output and pass through fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model