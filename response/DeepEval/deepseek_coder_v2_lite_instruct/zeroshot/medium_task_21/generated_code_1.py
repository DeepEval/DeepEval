import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, concatenate

def dl_model():
    # Define input shape
    input_shape = (32, 32, 3)
    
    # Define the input tensor
    inputs = Input(shape=input_shape)
    
    # Branch 1: 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    
    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    
    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch4 = Conv2D(32, (1, 1), activation='relu')(branch4)
    
    # Apply dropout to each branch
    branch1 = Dropout(0.25)(branch1)
    branch2 = Dropout(0.25)(branch2)
    branch3 = Dropout(0.25)(branch3)
    branch4 = Dropout(0.25)(branch4)
    
    # Concatenate the outputs from all branches
    combined = concatenate([branch1, branch2, branch3, branch4])
    
    # Flatten the concatenated output
    flattened = Flatten()(combined)
    
    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(64, activation='relu')(fc1)
    outputs = Dense(10, activation='softmax')(fc2)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()