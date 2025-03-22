import numpy as np
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Concatenate, Reshape, Conv2D
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for 28x28 grayscale images
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Parallel paths with different average pooling sizes
    path1 = AveragePooling2D(pool_size=(1, 1))(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4))(input_layer)
    
    # Flatten the outputs of each path
    flat1 = Flatten()(path1)
    flat2 = Flatten()(path2)
    flat3 = Flatten()(path3)
    
    # Concatenate the flattened outputs
    concat1 = Concatenate()([flat1, flat2, flat3])
    
    # Fully connected layer
    fc1 = Dense(128, activation='relu')(concat1)
    
    # Reshape for Block 2
    reshape_layer = Reshape((4, 4, 8))(fc1)  # Adjust the shape according to your need
    
    # Block 2: Multiple branches for feature extraction
    branch1 = Conv2D(32, (1, 1), activation='relu')(reshape_layer)
    branch1 = Conv2D(64, (3, 3), activation='relu')(branch1)

    branch2 = Conv2D(32, (1, 1), activation='relu')(reshape_layer)
    branch2 = Conv2D(32, (1, 7), activation='relu')(branch2)
    branch2 = Conv2D(32, (7, 1), activation='relu')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)

    branch3 = AveragePooling2D(pool_size=(2, 2))(reshape_layer)
    
    # Concatenate the outputs from all branches
    concat2 = Concatenate()([branch1, branch2, branch3])
    
    # Fully connected layers for final classification
    flatten_output = Flatten()(concat2)
    fc2 = Dense(128, activation='relu')(flatten_output)
    output_layer = Dense(10, activation='softmax')(fc2)  # 10 classes for MNIST
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example of using the model
model = dl_model()
model.summary()