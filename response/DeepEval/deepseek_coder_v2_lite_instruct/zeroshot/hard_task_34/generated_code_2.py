import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, Add, Flatten, Dense, SeparableConv2D

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Define the main path
    def main_path_block(x):
        x = SeparableConv2D(64, (3, 3), padding='same')(x)
        x = ReLU()(x)
        return x
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Main path
    x = main_path_block(inputs)
    x = main_path_block(x)
    x = main_path_block(x)
    
    # Branch path
    branch = Conv2D(64, (1, 1), padding='same')(inputs)
    branch = ReLU()(branch)
    
    # Fusion path
    x = Add()([x, branch])
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layer
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()