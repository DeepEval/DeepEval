import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Dropout, Flatten, Dense, Concatenate

def specialized_block(inputs):
    # Depthwise separable convolutional layer
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(inputs)
    x = Conv2D(filters=32, kernel_size=(1, 1))(x)
    x = Dropout(0.2)(x)
    
    return x

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Three branches
    branch1 = specialized_block(inputs)
    branch2 = specialized_block(inputs)
    branch3 = specialized_block(inputs)
    
    # Concatenate outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Flatten the concatenated output
    flattened = Flatten()(concatenated)
    
    # Fully connected layers
    x = Dense(128, activation='relu')(flattened)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()