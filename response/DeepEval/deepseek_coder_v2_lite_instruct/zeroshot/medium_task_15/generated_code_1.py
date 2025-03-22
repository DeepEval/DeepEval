import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D, Flatten

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Convolutional layer
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(inputs)
    
    # Batch Normalization and ReLU activation
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Compress feature maps using Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Two fully connected layers
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    
    # Reshape to match the channels of the initial features
    x = Reshape((1, 1, 64))(x)
    
    # Multiply with initial features
    x = Multiply()([x, inputs])
    
    # Concatenate with input layer
    x = Concatenate()([inputs, x])
    
    # Reduce dimensionality and downsample feature using 1x1 convolution and average pooling
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
    x = AveragePooling2D(pool_size=(8, 8))(x)
    
    # Flatten the output for the final fully connected layer
    x = Flatten()(x)
    
    # Output layer
    outputs = Dense(units=10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()