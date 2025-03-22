import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Dropout, Flatten, Dense

def dl_model():
    input_shape = (28, 28, 1)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Depthwise separable convolutional layer
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    
    # 1x1 convolutional layer for feature extraction
    x = Conv2D(filters=32, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layer
    outputs = Dense(units=10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()