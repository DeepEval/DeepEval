import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 Convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_layer)
    
    # 3x3 Depthwise Separable Convolutional layer for feature extraction
    depthwise_conv = SeparableConv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv1)
    
    # Another 1x1 Convolutional layer to reduce dimensionality
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(depthwise_conv)
    
    # Flattening layer
    flatten_layer = Flatten()(conv2)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model