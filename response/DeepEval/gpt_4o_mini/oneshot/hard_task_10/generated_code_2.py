import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, BatchNormalization

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First path: 1x1 convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second path: sequence of convolutions 1x1 -> 1x7 -> 7x1
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    # Concatenate the outputs of the two paths
    concatenated = keras.layers.Concatenate()([path1, path2])

    # 1x1 convolution to align output dimensions with input channels
    main_output = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch that connects directly to the input
    merged_output = Add()([main_output, input_layer])

    # Flatten and apply fully connected layers
    flatten_layer = Flatten()(merged_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model