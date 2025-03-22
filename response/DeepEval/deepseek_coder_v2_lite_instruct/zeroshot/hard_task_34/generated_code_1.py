import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, Add, Flatten, Dense, Concatenate
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Define the main path
    def main_path_block(x):
        x = SeparableConv2D(32, (3, 3), padding='same')(x)
        x = ReLU()(x)
        return x
    
    # Create the main path
    main_input = Input(shape=input_shape)
    x = main_input
    for _ in range(3):
        x = main_path_block(x)
    
    # Define the branch path
    branch_input = Conv2D(32, (1, 1), activation='relu')(main_input)
    
    # Concatenate the main path and branch path along the channel dimension
    combined = Add()([x, branch_input])
    
    # Flatten the combined features
    flattened = Flatten()(combined)
    
    # Pass through a fully connected layer to output the classification results
    outputs = Dense(10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=main_input, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()