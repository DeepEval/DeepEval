import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: 1x1 convolution
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: 1x1 -> 3x3 -> 3x3 convolutions
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_2)

    # Path 3: 1x1 -> 3x3 convolution
    conv3_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)

    # Path 4: Max pooling -> 1x1 convolution
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    conv4_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool4)

    # Concatenate all paths
    combined = Concatenate()([conv1_1, conv2_3, conv3_2, conv4_1])

    # Flatten and dense layer
    flatten = Flatten()(combined)
    dense = Dense(units=128, activation='relu')(flatten)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model