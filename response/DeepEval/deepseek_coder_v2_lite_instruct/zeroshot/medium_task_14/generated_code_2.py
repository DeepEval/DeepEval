import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Block 1
    x1 = Conv2D(32, (3, 3), padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv2D(32, (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv2D(32, (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    
    # Block 2
    x2 = Conv2D(64, (3, 3), padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = Conv2D(64, (3, 3), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = Conv2D(64, (3, 3), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    
    # Block 3
    x3 = Conv2D(128, (3, 3), padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    x3 = Conv2D(128, (3, 3), padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    x3 = Conv2D(128, (3, 3), padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    
    # Parallel branch
    y = Conv2D(32, (3, 3), padding='same')(inputs)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv2D(64, (3, 3), padding='same')(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv2D(128, (3, 3), padding='same')(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    
    # Addition of all paths
    combined = Add()([x1, x2, x3, y])
    
    # Flatten and fully connected layers
    z = Flatten()(combined)
    z = Dense(256, activation='relu')(z)
    outputs = Dense(10, activation='softmax')(z)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()