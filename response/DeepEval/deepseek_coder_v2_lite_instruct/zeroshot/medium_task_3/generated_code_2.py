import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Define the inputs
    inputs = Input(shape=input_shape)
    
    # First path
    conv1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    # Second path
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    # Combine the paths
    combined = Add()([pool2, inputs])
    
    # Flatten the combined output
    flattened = Flatten()(combined)
    
    # Fully connected layer
    outputs = Dense(10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()