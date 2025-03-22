import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Initial convolution
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # First block
    x1 = Conv2D(32, (3, 3), padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    
    # Second block
    x2 = Conv2D(32, (3, 3), padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    
    # Third block
    x3 = Conv2D(32, (3, 3), padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    
    # Add outputs of the blocks to the initial convolution
    x = Add()([x, x1, x2, x3])
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model