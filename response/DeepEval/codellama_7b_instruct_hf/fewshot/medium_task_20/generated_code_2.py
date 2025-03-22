import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # 1x1 convolutional path
    conv1_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # 3x3 convolutional path
    conv2_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)

    # 1x1 convolutional path
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Max pooling path
    maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    conv4_1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool_1)

    # Concatenate the outputs from all paths
    concatenated_output = Concatenate()([conv1_1, conv2_2, conv3_1, conv4_1])

    # Flatten and pass through dense layer
    flattened_output = Flatten()(concatenated_output)
    dense_output = Dense(units=128, activation='relu')(flattened_output)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model