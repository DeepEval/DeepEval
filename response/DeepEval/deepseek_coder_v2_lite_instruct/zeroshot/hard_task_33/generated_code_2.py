import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense, Concatenate

def same_block(input_layer):
    # 1x1 convolutional layer to elevate the dimension
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    # 3x3 depthwise separable convolutional layer
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)
    # 1x1 convolutional layer to reduce the dimension
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
    # Add the block's input to the output
    output_layer = Add()([input_layer, x])
    return output_layer

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
    
    # Fully connected layer to generate classification probabilities
    output_layer = Dense(10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()