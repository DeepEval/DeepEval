import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(x, filters):
    # Main path
    x1 = Conv2D(filters, (3, 3), padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    
    # Branch path
    branch = Conv2D(filters, (1, 1), padding='same')(x)
    
    # Add main path and branch path
    x2 = Add()([x1, branch])
    x2 = ReLU()(x2)
    
    return x2

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    
    # First basic block
    x = basic_block(x, 16)
    
    # Second basic block
    x = basic_block(x, 32)
    
    # Average pooling layer
    x = AveragePooling2D(pool_size=(8, 8))(x)
    
    # Flatten the feature map
    x = Flatten()(x)
    
    # Fully connected layer
    outputs = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()