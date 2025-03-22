import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def same_block(input_layer):
    # 1x1 convolutional layer to elevate the dimension
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # 1x1 convolutional layer to reduce the dimension
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
    
    # Add the block's input to the output
    output = Add()([input_layer, x])
    
    return output

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Create three branches
    branch1 = same_block(inputs)
    branch2 = same_block(inputs)
    branch3 = same_block(inputs)
    
    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Flatten the concatenated output
    flattened = Flatten()(concatenated)
    
    # Fully connected layer to generate classification probabilities
    outputs = Dense(10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()