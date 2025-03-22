import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, DepthwiseConv2D
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer to increase dimensionality
    conv1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer for feature extraction
    depthwise_conv3x3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1_1)
    
    # 1x1 convolutional layer to reduce dimensionality
    conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv3x3)
    
    # Add the output from the 1x1 convolutional layer to the original input layer
    added = Add()([input_layer, conv1x1_2])
    
    # Flatten the output
    flattened = Flatten()(added)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model