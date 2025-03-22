import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28,28,1))

    # 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer for feature extraction
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # 1x1 convolutional layer to reduce dimensionality
    conv3 = Conv2D(filters=10, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(conv2)

    # Flattening layer to prepare output for fully connected layer
    flatten_layer = Flatten()(conv3)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model