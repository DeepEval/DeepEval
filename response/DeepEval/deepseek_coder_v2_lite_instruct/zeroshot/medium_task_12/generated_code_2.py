import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Block 1
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x1 = x
    
    # Block 2
    x = Conv2D(64, (3, 3), padding='same')(x1)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x2 = x
    
    # Block 3
    x = Conv2D(128, (3, 3), padding='same')(x2)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x3 = x
    
    # Concatenate outputs of the blocks along the channel dimension
    x = Add()([x1, x2, x3])
    
    # Flatten the concatenated output
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()