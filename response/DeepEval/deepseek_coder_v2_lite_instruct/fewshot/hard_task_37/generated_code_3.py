import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv1, conv2, conv3

    # First block
    conv1_1, conv2_1, conv3_1 = block(input_layer)
    main_path_1 = Add()([conv1_1, conv2_1, conv3_1])

    # Second block
    conv1_2, conv2_2, conv3_2 = block(input_layer)
    main_path_2 = Add()([conv1_2, conv2_2, conv3_2])

    # Concatenate outputs from both blocks
    concatenated = Concatenate()([main_path_1, main_path_2])

    # Flatten and fully connected layer
    flattened = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model