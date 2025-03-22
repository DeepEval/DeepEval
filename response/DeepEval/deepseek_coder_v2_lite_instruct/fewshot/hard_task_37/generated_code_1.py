import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate

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

    # Branch path
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine outputs from all paths
    adding_layer = Add()([main_path_1, main_path_2, branch_path])

    # Concatenate outputs from the two blocks
    concatenate_layer = Concatenate()([main_path_1, main_path_2])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(concatenate_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model