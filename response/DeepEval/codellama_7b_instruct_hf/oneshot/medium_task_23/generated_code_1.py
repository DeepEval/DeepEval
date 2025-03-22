import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: single 1x1 convolution
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    conv2 = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Path 3: 1x1 convolution followed by a combination of two sets of 1x7 and 7x1 convolutions to capture multi-directional spatial features
    conv4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5 = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv5)

    # Path 4: average pooling followed by a 1x1 convolution
    pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    conv7 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool)

    # Fuse outputs of paths through concatenation
    outputs = Concatenate()([conv1, conv2, conv3, conv4, conv5, conv6, conv7])

    # Flatten output and pass through fully connected layers for classification
    flatten = Flatten()(outputs)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model