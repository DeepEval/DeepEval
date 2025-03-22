import keras
from keras.layers import Input, Conv2D, Add, Dense, Flatten, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First path: 1x1 convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second path: 1x1 -> 1x7 -> 7x1 convolutions
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    # Concatenate paths
    concatenated = Concatenate()([path1, path2])

    # 1x1 convolution to align dimensions
    main_path_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch that directly connects to input
    branch = input_layer

    # Add main path output and branch
    added = Add()([main_path_output, branch])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(added)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model