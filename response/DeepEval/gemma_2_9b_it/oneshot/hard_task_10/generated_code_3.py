import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Path 1: 1x1 Convolution
    conv1_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: 1x1, 1x7, 7x1 Convolutions
    conv2_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_path = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv2_path)
    conv2_path = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv2_path)

    # Concatenate outputs of both paths
    concat_layer = Concatenate()([conv1_path, conv2_path])

    # 1x1 Convolution to align dimensions
    main_path_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)

    # Direct branch connection
    branch_output = input_layer

    # Add main path output and branch output
    final_output = keras.layers.Add()([main_path_output, branch_output])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(final_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model