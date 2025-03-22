import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Conv2DTranspose, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # 3x3 depthwise separable convolutional layer for feature extraction
    dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv1)
    dw_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(dw_conv)

    # 1x1 convolutional layer to reduce dimensionality with a stride of 2
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(dw_conv)

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(conv2)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model