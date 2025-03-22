import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, DepthwiseConv2D, Lambda, Flatten, Dense

def dl_model():

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # 3x3 depthwise separable convolutional layer for feature extraction
    dw_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # 1x1 convolutional layer to reduce dimensionality with stride of 2
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(dw_conv1)

    # Average pooling with a pool size of 2x2 and stride of 2
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)

    # Output layer for classification probabilities
    flatten = Flatten()(avg_pool)
    dense = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model