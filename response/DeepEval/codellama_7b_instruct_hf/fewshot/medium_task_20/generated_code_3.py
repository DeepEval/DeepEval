import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First path: 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second path: 3x3 convolutions stacked after a 1x1 convolution
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Third path: single 3x3 convolution followed by a 1x1 convolution
    conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4)

    # Fourth path: max pooling followed by a 1x1 convolution
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    conv6 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool)

    # Concatenate outputs from all paths
    concatenated = Concatenate()([conv1, conv2, conv3, conv4, conv5, conv6])

    # Flatten and pass through a dense layer
    flattened = Flatten()(concatenated)
    dense = Dense(units=128, activation='relu')(flattened)

    # Output layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model