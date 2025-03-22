import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 Convolution
    branch1_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: 1x1 + 3x3 Convolution
    branch2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_conv1)

    # Branch 3: 1x1 + 5x5 Convolution
    branch3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3_conv2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch3_conv1)

    # Branch 4: 3x3 Max Pooling + 1x1 Convolution
    branch4_maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch4_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4_maxpool)

    # Concatenate branches
    concat = Concatenate()([branch1_conv, branch2_conv2, branch3_conv2, branch4_conv])

    # Flatten and fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model