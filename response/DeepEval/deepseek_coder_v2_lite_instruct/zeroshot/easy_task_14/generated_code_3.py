import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First convolutional layer
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(x)
    
    # Fully connected layer to generate weights
    fc1 = Dense(128, activation='relu')(gap)
    weights = Dense(input_shape[2], activation='sigmoid')(fc1)
    
    # Reshape weights to align with input shape
    reshaped_weights = Reshape((1, 1, input_shape[2]))(weights)
    
    # Multiply element-wise with the input feature map
    multiplied = Multiply()([x, reshaped_weights])
    
    # Flatten the result
    flattened = Flatten()(multiplied)
    
    # Final fully connected layer
    outputs = Dense(10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()