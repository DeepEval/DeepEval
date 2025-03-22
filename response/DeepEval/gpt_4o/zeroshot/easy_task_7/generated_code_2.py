import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape based on MNIST image dimensions
    input_shape = (28, 28, 1)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Main path
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)  # Restore the number of channels to 1
    
    # Branch path
    branch = inputs  # Direct connection to the input
    
    # Combine outputs from both paths
    combined = Add()([x, branch])
    
    # Flatten and output layer
    flat = Flatten()(combined)
    outputs = Dense(10, activation='softmax')(flat)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()