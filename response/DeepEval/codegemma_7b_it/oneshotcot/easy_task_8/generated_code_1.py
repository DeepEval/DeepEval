import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Depthwise Separable Convolutional Layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    depthwise_conv = Dropout(rate=0.2)(depthwise_conv)

    # 1x1 Convolutional Layer for Feature Extraction
    pointwise_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    pointwise_conv = Dropout(rate=0.2)(pointwise_conv)

    # Flattening and Fully Connected Layer for Classification
    flatten_layer = Flatten()(pointwise_conv)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model