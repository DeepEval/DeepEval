import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, DepthwiseConv2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer for feature extraction
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Second 1x1 convolutional layer to reduce dimensionality
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    
    # Add the output from the second 1x1 convolutional layer to the original input layer
    added = Add()([conv2, input_layer])
    
    # Flatten the output
    flatten_layer = Flatten()(added)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model