import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Add, Flatten, Dense, Concatenate

def same_block(input_layer):
    # 1x1 Convolution to increase the dimension
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # 3x3 Depthwise Convolution
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # 1x1 Convolution to reduce the dimension
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    
    # Add the block's input to the output
    x = Add()([x, input_layer])
    
    return x

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Three branches
    branch1 = same_block(input_layer)
    branch2 = same_block(input_layer)
    branch3 = same_block(input_layer)
    
    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Flatten the concatenated output
    flattened = Flatten()(concatenated)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()