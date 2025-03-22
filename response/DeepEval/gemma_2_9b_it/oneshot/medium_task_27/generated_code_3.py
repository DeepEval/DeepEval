import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutions
    conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)

    # Branch 2: 5x5 convolutions
    conv2_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2_1)

    # Combine branches using addition
    combined = Concatenate()([conv1_2, conv2_2])

    # Global average pooling
    pool = AveragePooling2D(pool_size=(8, 8), strides=(8, 8))(combined)
    flatten = Flatten()(pool)

    # Attention weights
    dense1 = Dense(units=10, activation='softmax')(flatten)

    # Weighted sum
    output = keras.layers.multiply([conv1_2, dense1]) + keras.layers.multiply([conv2_2, 1 - dense1])

    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model