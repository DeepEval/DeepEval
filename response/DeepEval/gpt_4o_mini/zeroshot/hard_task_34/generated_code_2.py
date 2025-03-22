import numpy as np
from keras.layers import Input, Conv2D, SeparableConv2D, ReLU, Add, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))  # MNIST images are 28x28 with 1 channel (grayscale)
    
    # Main path
    main_path = input_layer
    for _ in range(3):  # Repeat the block three times
        # Separable Convolution block
        x = SeparableConv2D(32, (3, 3), padding='same')(main_path)
        x = ReLU()(x)
        main_path = Concatenate()([main_path, x])  # Concatenate the input with the output of the conv layer
    
    # Branch path
    branch_path = Conv2D(32, (3, 3), padding='same')(input_layer)
    
    # Fusion of the two paths
    fused = Add()([main_path, branch_path])
    
    # Flatten the features
    flattened = Flatten()(fused)
    
    # Fully connected layer for classification
    output_layer = Dense(10, activation='softmax')(flattened)  # 10 classes for digits 0-9
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example of how to create the model
model = dl_model()
model.summary()  # Print the model summary to verify the architecture