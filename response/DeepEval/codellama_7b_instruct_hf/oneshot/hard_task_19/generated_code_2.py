import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Main path
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Branch path
    branch_input = Input(shape=(32, 32, 3))
    gavg = GlobalAveragePooling2D()(branch_input)
    fc1 = Dense(units=64, activation='relu')(gavg)
    fc2 = Dense(units=32, activation='relu')(fc1)
    channel_weights = Dense(units=128, activation='softmax')(fc2)
    channel_weights = Flatten()(channel_weights)
    channel_weights = Reshape((128, 1))(channel_weights)

    # Concat and add
    main_path_output = Concatenate()([maxpool, channel_weights])
    main_path_output = BatchNormalization()(main_path_output)
    main_path_output = Flatten()(main_path_output)
    main_path_output = Dense(units=128, activation='relu')(main_path_output)
    main_path_output = Dense(units=64, activation='relu')(main_path_output)
    main_path_output = Dense(units=10, activation='softmax')(main_path_output)

    # Output layer
    output_layer = Concatenate()([main_path_output, branch_output])
    output_layer = Flatten()(output_layer)
    output_layer = Dense(units=128, activation='relu')(output_layer)
    output_layer = Dense(units=64, activation='relu')(output_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=[input_layer, branch_input], outputs=output_layer)

    return model