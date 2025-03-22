import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer to increase dimensionality
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer for feature extraction
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # Another 1x1 convolutional layer to reduce dimensionality
    x = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x)
    
    # Add the processed output to the original input layer
    x = Add()([x, input_layer])
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layer to generate the final classification probabilities
    output_layer = Dense(units=10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model