import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # 3x3 depthwise separable convolutional layer for feature extraction
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # 1x1 convolutional layer to reduce dimensionality
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)

    # MaxPooling2D with stride of 2
    pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Flatten the output
    flatten = Flatten()(pool)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model